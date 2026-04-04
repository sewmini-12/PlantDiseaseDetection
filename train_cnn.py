import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cnn import create_cnn_model
import numpy as np
import time

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'split_data/train'
TEST_DIR = 'split_data/test'

def main():
    print("Initializing PyTorch for CNN training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(train_data.classes)
    print(f"Data loaded: {len(train_data)} train images, {len(test_data)} test images.")

    model = create_cnn_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting CNN training (PyTorch)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - time: {time.time()-start_time:.1f}s")

    print("\nEvaluating CNN model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"CNN Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), 'plant_disease_cnn.pth')
    np.save('cnn_final_acc.npy', np.array([accuracy]))
    print("CNN Results saved. Ready for Comparison.")

if __name__ == "__main__":
    main()
