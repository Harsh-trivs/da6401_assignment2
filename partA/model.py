import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
from sklearn.model_selection import StratifiedShuffleSplit

class CustomCNN(nn.Module):
    
    def __init__(self,num_filters,hidden_size,filter_size=3,num_classes=10,activation_function='ReLU',n_blocks=5,filter_organisation_factor=2,dropout=0.4,batch_norm=True):
        """
        Custom CNN model for image classification.
        Args:
            num_filters (int): Number of filters in the first convolutional layer.
            size_filter (int): Size of the convolutional filters.
            num_classes (int): Number of output classes.
            activation (str): Activation function to use .
            n_blocks (int): Number of convolutional blocks.
        """
        super(CustomCNN,self).__init__()
        self.conv = nn.ModuleList()
        self.activation = {"ReLU": nn.ReLU(), "GeLU": nn.GELU(), "SiLU": nn.SiLU(), "Mish": nn.Mish()}
        in_channel = 3
        # convolutional blocks
        for i in range(n_blocks):
            in_channel = num_filters*(filter_organisation_factor**(i-1)) if i>0 else 3
            in_channel = int(in_channel)
            numberOfFilters = num_filters*(filter_organisation_factor**(i))
            numberOfFilters = int(numberOfFilters)
            # start of the convolutional block
            # convolution layer
            self.conv.append(nn.Conv2d(in_channels = in_channel, out_channels=numberOfFilters, kernel_size=filter_size, stride=1))
            # batch normalization layer
            if batch_norm:
                self.conv.append(nn.BatchNorm2d(numberOfFilters))
            # activation layer
            self.conv.append(self.activation[activation_function])
            # max pooling layer
            self.conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # flattening last layer
        self.conv.append(nn.AdaptiveAvgPool2d(1))
        self.conv.append(nn.Flatten())
        # fully connected layer
        in_channel = num_filters*(filter_organisation_factor**(n_blocks-1))
        in_channel = int(in_channel)    
        self.conv.append(nn.Linear(in_channel, hidden_size))
        self.conv.append(self.activation['GeLU'])
        # dropout layer
        self.conv.append(nn.Dropout(dropout))
        self.conv.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv:
            x = layer(x)
        return x
    
    def train_model(self, 
                   train_loader: DataLoader, 
                   val_loader: DataLoader, 
                   epochs: int, 
                   learning_rate: float, 
                   device: torch.device,
                   criterion: nn.Module = nn.CrossEntropyLoss(),
                   optimizer_class: optim.Optimizer = optim.Adam):
        """
        Train the model with accuracy evaluation.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimizer.
            device (torch.device): Device to train on ('cuda' or 'cpu').
            criterion (nn.Module): Loss function.
            optimizer_class (optim.Optimizer): Optimizer class (e.g., Adam, SGD).
        Returns:
            Dict[str, List[float]]: Dictionary containing training/validation losses and accuracies.
        """
        self.to(device)
        optimizer = optimizer_class(self.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_correct = 0
            total_samples = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_size
                running_correct += (outputs.argmax(1) == labels).sum().item()
 
                del loss
                del inputs
            # Calculate training metrics
            epoch_train_loss = running_loss / total_samples
            epoch_train_acc = running_correct / total_samples
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_size = inputs.size(0)
                    val_total += batch_size
                    
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * batch_size
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    del loss
                    del inputs
            # Calculate validation metrics
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            # Print metrics
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | '
                  f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')
        
 
        return history
    
    def predict(self, 
                test_loader: DataLoader, 
                device: torch.device) :
        """
        Make predictions on test data.
        Args:
            test_loader (DataLoader): DataLoader for test data.
            device (torch.device): Device to use for prediction.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions and true labels.
        """
        self.eval()
        self.to(device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        return torch.cat(all_preds), torch.cat(all_labels)

def transform_image(dataAugmentation=False):
    if dataAugmentation:
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize to 224x224
            transforms.RandomRotation(15), # Random rotation
            transforms.RandomHorizontalFlip(), # Random horizontal flip
            transforms.ToTensor(), # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],       
                std=[0.229, 0.224, 0.225])
        ]) # Normalize with ImageNet stats
        return transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],       
            std=[0.229, 0.224, 0.225]
        ) # Normalize with ImageNet stats
    ])
    return transform

def data_loader(data_dir, batch_size, dataAugmentation, num_workers=3):
    # Load the full training dataset
    full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_image(dataAugmentation=dataAugmentation))
    targets = full_dataset.targets  # class labels for stratification

    # Stratified split: 80% train, 20% val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(full_dataset.samples, targets))

    # Subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Override transform for val set (no augmentation)
    val_dataset.dataset.transform = transform_image(dataAugmentation=False)

    # Test dataset
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_image(dataAugmentation=False))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
