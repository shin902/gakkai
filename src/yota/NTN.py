import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Double Convolution block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# UNet Model
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        enc1 = self.encoder1(x)
        p1 = F.max_pool2d(enc1, 2)
        enc2 = self.encoder2(p1)
        p2 = F.max_pool2d(enc2, 2)
        enc3 = self.encoder3(p2)
        p3 = F.max_pool2d(enc3, 2)
        enc4 = self.encoder4(p3)
        p4 = F.max_pool2d(enc4, 2)
        bottleneck = self.bottleneck(p4)
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat([d4, F.interpolate(enc4, size=d4.shape[2:])], dim=1)
        d4 = self.decoder4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, F.interpolate(enc3, size=d3.shape[2:])], dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, F.interpolate(enc2, size=d2.shape[2:])], dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, F.interpolate(enc1, size=d1.shape[2:])], dim=1)
        d1 = self.decoder1(d1)
        output = self.final_conv(d1)
        return torch.sigmoid(F.interpolate(output, size=input_size, mode='bilinear', align_corners=False))

# Dataset for noisy image pairs
class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_pairs = []
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                noisy_images = sorted(list(folder.glob('*.jpg')))
                for i in range(len(noisy_images) - 1):
                    self.image_pairs.append((noisy_images[i], noisy_images[i + 1]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        input_img = Image.open(str(img1_path)).convert('RGB')
        target_img = Image.open(str(img2_path)).convert('RGB')
        input_img = input_img.resize((780, 972), Image.BILINEAR)
        target_img = target_img.resize((780, 972), Image.BILINEAR)
        return self.transform(input_img), self.transform(target_img)

# Training class
class Noise2Noise:
    def __init__(self, train_dir, valid_dir, model_dir, device):
        self.device = device
        self.model = UNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.train_dataset = NoisyDataset(train_dir)
        self.valid_dataset = NoisyDataset(valid_dir)
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for input_img, target_img in self.train_loader:
                input_img, target_img = input_img.to(self.device), target_img.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(input_img), target_img)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{epochs} completed.')

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

    def denoise_image(self, input_path, output_path):
        self.model.eval()
        input_img = Image.open(input_path).convert('RGB')
        input_tensor = transforms.ToTensor()(input_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_img.save(output_path)

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Noise2Noise("../../Resources/AI/train_data", "../../Resources/AI/valid_data", "../../Resources/AI/model_dir", device)
    trainer.load_model("../../Resources/AI/model_dir/best_model.pth")
    trainer.denoise_image("0824.bmp", "output.jpg")
