import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from network import VisualDriveNet
import numpy as np
import matplotlib.pyplot as plt

class F1Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'image': row['image_path'],
                    'steering': float(row['angular_w'])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['image'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        steering = self.data[idx]['steering']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([steering], dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"hardware locked to: {device}")

    dataset_dir = os.path.expanduser('~/jderobot_gsoc26/dataset')
    csv_path = os.path.join(dataset_dir, 'driving_log.csv')
    images_dir = os.path.join(dataset_dir, 'images')
    model_dir = os.path.expanduser('~/jderobot_gsoc26/src/visual_control_module/models')
    
    model_save_path = os.path.join(model_dir, 'robust_model.pth')
    plot_save_path = os.path.join(model_dir, 'robust_training_loss.png')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"retrieving dataset from {images_dir}")
    f1_dataset = F1Dataset(csv_file=csv_path, root_dir=images_dir, transform=transform)
    dataloader = DataLoader(f1_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = VisualDriveNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    loss_history = []
    print("training sequence initiated")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, steerings) in enumerate(dataloader):
            images, steerings = images.to(device), steerings.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, steerings)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"epoch {epoch+1} completed | avg loss: {epoch_loss:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"robust model successfully saved to {model_save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='#9b59b6', label='Robust Training Loss')
    plt.title('Robust VisualDriveNet Convergence', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"loss visualization saved to {plot_save_path}")

if __name__ == '__main__':
    main()