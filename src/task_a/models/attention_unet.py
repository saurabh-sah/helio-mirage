import torch
import torch.nn as nn
import torch.nn.functional as F
from .vanilla_unet import DoubleConv

class AttentionBlock(nn.Module):
    """Attention block to focus on relevant features."""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpAttention(nn.Module):
    """Upscaling then attention then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpAttention, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        x2 = self.attention(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUNet(nn.Module):
    """
    Attention U-Net with Deep Supervision and Skeleton Auxiliary Head.
    As per Phase 3, this model uses attention to reject background noise,
    and a skeleton auxiliary head + deep supervision to preserve thin barb structures.
    """
    def __init__(self, img_ch=1, output_ch=1):
        super(AttentionUNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(img_ch, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.Up5 = UpAttention(1024, 512, bilinear=False)
        self.Up4 = UpAttention(512, 256, bilinear=False)
        self.Up3 = UpAttention(256, 128, bilinear=False)
        self.Up2 = UpAttention(128, 64, bilinear=False)

        # Main output head (Filament Segmentation)
        self.out_main = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        # Skeleton Auxiliary Head (Supervises thin structures specifically)
        self.out_skeleton = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        # Deep Supervision Heads
        self.ds3 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.ds2 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        # Decoder with Attention
        d5 = self.Up5(e5, e4)
        d4 = self.Up4(d5, e3)
        d3 = self.Up3(d4, e2)
        d2 = self.Up2(d3, e1)

        # Outputs
        out_main = self.out_main(d2)
        out_skeleton = self.out_skeleton(d2)
        
        # Deep Supervision Outputs
        out_ds3 = self.ds3(d4)
        out_ds2 = self.ds2(d3)
        
        # Resize deep supervision outputs to match main output resolution
        out_ds3 = F.interpolate(out_ds3, size=out_main.shape[2:], mode='bilinear', align_corners=False)
        out_ds2 = F.interpolate(out_ds2, size=out_main.shape[2:], mode='bilinear', align_corners=False)

        if self.training:
            return out_main, out_skeleton, out_ds3, out_ds2
        else:
            return out_main
