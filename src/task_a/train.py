import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.task_a.datasets.magfilo import MAGFiLODataset
from src.task_a.models.attention_unet import AttentionUNet
from src.task_a.losses.cldice import CombinedLoss

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        spines = batch['spine'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        out_main, out_skeleton, out_ds3, out_ds2 = model(images)
        
        # Losses
        loss_main = criterion(out_main, masks, epoch)
        loss_skel = criterion(out_skeleton, spines, epoch) # using spines for aux head
        loss_ds3 = criterion(out_ds3, masks, epoch)
        loss_ds2 = criterion(out_ds2, masks, epoch)
        
        # Deep supervision weights
        total_loss = loss_main + 0.5 * loss_skel + 0.3 * loss_ds3 + 0.2 * loss_ds2
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        pbar.set_postfix({'loss': total_loss.item()})
        
    return running_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4
    epochs = 50
    lr = 1e-4
    
    # Dataset and DataLoader
    train_dataset = MAGFiLODataset(root_dir='data/processed', split='train', patch_size=256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = AttentionUNet(img_ch=1, output_ch=1).to(device)
    
    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = CombinedLoss(alpha=0.5, bce_weight=0.5)
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.4f}")
        
        if epoch % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/task_a_epoch_{epoch}.pth")
            
if __name__ == "__main__":
    main()
