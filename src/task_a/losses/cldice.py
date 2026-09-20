import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(img):
    """Soft erosion using max pooling on inverted image."""
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1,p2)

def soft_dilate(img):
    """Soft dilation using max pooling."""
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    """Soft morphological opening."""
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, iter_=10):
    """
    Computes a soft skeleton of the image.
    Uses iterative morphological opening.
    """
    img1 = img
    img2 = soft_erode(img1)
    img3 = soft_open(img2)
    img_skel = F.relu(img1 - img3)
    
    for i in range(iter_):
        img1 = img2
        img2 = soft_erode(img1)
        img3 = soft_open(img2)
        img_skel = img_skel + F.relu(img1 - img3)
        
    return img_skel

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dice

class clDiceLoss(nn.Module):
    """
    clDice Loss for segmenting tubular/thin structures.
    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
    Where Tprec is soft-skeleton of prediction intersecting GT mask,
    and Tsens is soft-skeleton of GT intersecting prediction mask.
    """
    def __init__(self, smooth=1e-5, iter_=10):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth
        self.iter_ = iter_

    def forward(self, y_pred, y_true):
        # Compute soft skeletons
        skel_pred = soft_skeletonize(y_pred, self.iter_)
        skel_true = soft_skeletonize(y_true, self.iter_)
        
        # Tprec: How much of the predicted skeleton lies within the true mask?
        tprec = (skel_pred * y_true).sum() / (skel_pred.sum() + self.smooth)
        
        # Tsens: How much of the true skeleton lies within the predicted mask?
        tsens = (skel_true * y_pred).sum() / (skel_true.sum() + self.smooth)
        
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        
        return 1.0 - cl_dice

class CombinedLoss(nn.Module):
    """
    Combines Dice, BCE, and clDice with a warmup schedule for clDice 
    (as clDice is unstable early in training).
    """
    def __init__(self, alpha=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.cldice = clDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha # weight for clDice (warmed up over epochs)
        self.bce_weight = bce_weight

    def forward(self, y_pred_logits, y_true, current_epoch, warmup_epochs=10):
        # 1. Standard BCE on logits
        loss_bce = self.bce(y_pred_logits, y_true)
        
        # 2. Dice and clDice require probabilities
        y_pred_probs = torch.sigmoid(y_pred_logits)
        
        loss_dice = self.dice(y_pred_probs, y_true)
        
        # 3. Calculate clDice (only apply if past warmup or warm it up linearly)
        cldice_weight = self.alpha * min(1.0, current_epoch / max(1, warmup_epochs))
        
        if cldice_weight > 0:
            loss_cldice = self.cldice(y_pred_probs, y_true)
        else:
            loss_cldice = 0.0
            
        total_loss = self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_dice + cldice_weight * loss_cldice
        
        return total_loss
