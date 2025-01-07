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

# UNet architecture
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
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
        # Encoder
        enc1 = self.encoder1(x)
        x = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(x)
        x = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(x)
        x = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(x)
        x = F.max_pool2d(enc4, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)

        return torch.sigmoid(self.final_conv(x))

# Dataset for noisy image pairs
class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_pairs = []

        # Find all image pairs
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                noisy_images = list(folder.glob('*.jpg'))
                if len(noisy_images) >= 2:
                    self.image_pairs.append((noisy_images[0], noisy_images[1]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]

        # Load images
        input_img = Image.open(img1_path).convert('RGB')
        target_img = Image.open(img2_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

# Training class
class Noise2Noise:
    def __init__(self, train_dir, valid_dir, model_dir, device):
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize model
        self.model = UNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Setup transforms
        self.transform = transforms.Compose([
            # transforms.Resize((800, 600)),
            transforms.ToTensor(),
        ])

        # Setup datasets
        self.train_dataset = NoisyDataset(train_dir, self.transform)
        self.valid_dataset = NoisyDataset(valid_dir, self.transform)

        # Setup dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, shuffle=False)

    def train(self, epochs):
        best_valid_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, (input_img, target_img) in enumerate(self.train_loader):
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(input_img)
                loss = self.criterion(output, target_img)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} [{batch_idx}/{len(self.train_loader)}] '
                          f'Loss: {loss.item():.6f}')

            # Validation
            valid_loss = self.validate()
            print(f'Epoch {epoch+1} Average Train Loss: {train_loss/len(self.train_loader):.6f} '
                  f'Valid Loss: {valid_loss:.6f}')

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model('best_model.pth')

    def validate(self):
        self.model.eval()
        valid_loss = 0

        with torch.no_grad():
            for input_img, target_img in self.valid_loader:
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                output = self.model(input_img)
                loss = self.criterion(output, target_img)
                valid_loss += loss.item()

        return valid_loss / len(self.valid_loader)

    def save_model(self, filename):
        save_path = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load_model(self, filename):
      load_path = self.model_dir / filename
      checkpoint = torch.load(load_path, weights_only=True)  # weights_only=True を追加
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def denoise_image(self, input_path, output_path):
        self.model.eval()

        # Load and transform image
        input_img = Image.open(input_path).convert('RGB')
        input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        # Convert output tensor to image
        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_img.save(output_path)

# Example usage
if __name__ == "__main__":
    # Setup device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # Initialize trainer
    trainer = Noise2Noise(
        train_dir="train_dir",
        valid_dir="valid_data",
        model_dir="model_dir",
        device=device
    )

    # Train model
    trainer.train(epochs=1000)
    # trainer.load_model('best_model.pth')

    # Denoise a single image
    trainer.denoise_image("input/111.jpg", "output/1000-1000-noclop.jpg")
