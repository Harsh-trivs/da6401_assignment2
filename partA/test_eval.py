import torch
import wandb
import os
from partA.model import CustomCNN, data_loader
import torchvision.datasets as datasets
import numpy as np

train_loader, val_loader, test_loader = data_loader(data_dir='data', batch_size=32, dataAugmentation=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomCNN(
                    filter_size=3,
                    num_classes=10,
                    num_filters=64,
                    hidden_size=256,
                    filter_organisation_factor=1.5,
                    dropout=0.4,
                    batch_norm=True,
                    n_blocks=5,
                    activation_function="GeLU").to(device)

# Check if the best model exists
if os.path.exists("best_model.pth"):
    print("Best model found. Skipping training...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
else:
    print("No best model found. Starting training...")
    history = model.train_model(train_loader=train_loader,
                                val_loader=val_loader,
                                epochs=15,
                                learning_rate=0.001,
                                device=device)
    # Save the best model
    torch.save(model.state_dict(), "best_model.pth")

model.load_state_dict(torch.load("best_model.pth"))
# ---- Get best epoch based on test accuracy ----
predictions,true_labels = model.predict(test_loader, device)
best_accuracy = (predictions == true_labels).sum().item() / len(true_labels)

dataset = datasets.ImageFolder(root=os.path.join('data', 'val'))
class_to_name = {i: name.split('/')[-1] for i, name in enumerate(dataset.classes)}
                               

# ---- Initialize Weights & Biases ----
wandb.init(project="iNaturalist-CNN", name="best_model_test_eval")

# ---- Get 30 sample predictions ----
model.eval()
images, labels, preds = [], [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)

        images.extend(inputs.cpu())
        labels.extend(targets.cpu())
        preds.extend(predicted.cpu())

        if len(images) >= 30:
            break

images = images[:30]
labels = labels[:30]
preds = preds[:30]

# ---- Create 10×3 prediction grid ----
wandb_images =[]
for i in range(30):
        # Denormalize image
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Create caption with prediction info
        caption = (f"Correct ✅" if labels[i] == preds[i] else f"Wrong ❌") + \
                 f"\nPredicted: {class_to_name[labels[i].item()]}\nActual: {class_to_name[preds[i].item()]}"
        
        # Create wandb.Image with caption
        wandb_images.append(wandb.Image(
            img, 
            caption=caption,
        ))

# ---- Log grid and accuracy to wandb ----
wandb.log({
    "Prediction Grid": wandb_images,
    "Best Test Accuracy": best_accuracy
})