import os
import sys

# Add project root to sys.path so we can import from 'app'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.ml.cnn_model import CropDiseaseModel

def train_model(data_dir="knowledge_base/plantvillage dataset/color", epochs=5, batch_size=32, save_path="app/ml/model_checkpoint.pth"):
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Cannot train.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Using Hybrid Architecture: VGG16 + Spatial Attention + MobileNetV2")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load dataset
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        num_classes = len(full_dataset.classes)
        print(f"Found {num_classes} classes.")
        
        # Split dataset deterministically into 80% train, 20% test
        generator = torch.Generator().manual_seed(42)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)
        
        print(f"Training on {len(train_dataset)} images.")
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Initialize the hybrid model
        model = CropDiseaseModel(num_classes=num_classes).to(device)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Resume from checkpoint if it exists
        if os.path.exists(save_path):
            try:
                print(f"Found existing checkpoint at {save_path}. Resuming training from these weights...")
                model.load_state_dict(torch.load(save_path, map_location=device))
            except Exception as e:
                print(f"Could not load checkpoint: {e}")

        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            print(f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            print(f"Loss: {epoch_loss:.4f}")
            
            # Save model after every epoch so progress is not lost!
            torch.save(model.state_dict(), save_path)
            print(f"Progress saved to {save_path} (Epoch {epoch+1}/{epochs})")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_model()
