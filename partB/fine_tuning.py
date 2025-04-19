from partA.model import data_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import wandb


# Configuration
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 15
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Init wandb
wandb.init(project="resnet50-finetune-inaturalist", config={
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "architecture": "resnet50",
    "unfrozen_layers": "layer4 + fc",
})


# Load Dataset
train_loader, val_loader, test_loader = data_loader(
    data_dir='data', batch_size=BATCH_SIZE, dataAugmentation=True
)


# Load Pretrained Model
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last conv block (layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)


# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Training & Evaluation
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(output, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    return running_loss / len(loader), acc


def validate(model, loader, desc="Validation"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            _, preds = torch.max(output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# Main Training Loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    val_acc = validate(model, val_loader, desc="Validation")
    test_acc = validate(model, test_loader, desc="Testing")

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "lr": scheduler.get_last_lr()[0]
    })

    scheduler.step()


# Save model
torch.save(model.state_dict(), "resnet50_inaturalist_finetuned.pth")
wandb.save("resnet50_inaturalist_finetuned.pth")
wandb.finish()
