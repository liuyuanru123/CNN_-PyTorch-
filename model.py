import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset(dataset_path):
    images, labels = [], []
    for people_count in sorted(os.listdir(dataset_path)):
        people_folder_path = os.path.join(dataset_path, people_count)
        if not os.path.isdir(people_folder_path):
            continue
            
        for time_folder in os.listdir(people_folder_path):
            time_folder_path = os.path.join(people_folder_path, time_folder)
            if not os.path.isdir(time_folder_path):
                continue
                
            image_path = os.path.join(time_folder_path, 'Pytorch_Architecture.png')
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('L')
                image = image.resize((64, 128))
                image = np.array(image)
                images.append(image)
                labels.append(int(people_count) - 1)
                
    return np.array(images), np.array(labels)

def train_model():
    # Create output directory
    now = datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset_path = "./dataset"
    images, labels = load_dataset(dataset_path)
    
    # Setup transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Setup k-fold cross validation
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    # Training settings
    num_epochs = 200
    batch_size = 32
    
    # Metrics storage
    class_acc_per_fold = np.zeros((num_folds, 6))
    acc_per_fold = []
    loss_per_fold = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(images)):
        print(f'FOLD {fold + 1}')
        
        # Create datasets for this fold
        train_dataset = ImageDataset(
            images[train_ids], 
            labels[train_ids],
            transform=train_transform
        )
        val_dataset = ImageDataset(
            images[val_ids],
            labels[val_ids],
            transform=val_transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model, loss, and optimizer
        model = CNNModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(os.path.join(folder_path, f'logs/fold_{fold + 1}'))
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_images)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            accuracy = 100. * correct / total
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(folder_path, f'model_fold_{fold + 1}.pth'))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        class_accuracy = np.trace(cm) / np.sum(cm)
        class_acc_per_fold[fold] = [class_accuracy] * 6
        acc_per_fold.append(accuracy)
        loss_per_fold.append(val_loss)
        
        writer.close()
    
    # Plot final results
    class_specific_avg_accuracies = np.mean(class_acc_per_fold, axis=0) * 100
    heatmap_matrix = np.zeros((6, 6))
    np.fill_diagonal(heatmap_matrix, class_specific_avg_accuracies)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=[f'Class {i}' for i in range(6)],
                yticklabels=[f'Class {i}' for i in range(6)])
    plt.title('Average Prediction Accuracy per Class')
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth')
    plt.savefig(os.path.join(folder_path, 'Average_Prediction_Accuracy_per_Class.png'))
    plt.close()

if __name__ == "__main__":
    train_model()
