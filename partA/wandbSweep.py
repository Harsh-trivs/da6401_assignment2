from partA.model import CustomCNN, data_loader
import torch
import wandb
import gc

def train():
    wandb.init()
    config = wandb.config
    train_loader, val_loader, test_loader = data_loader(data_dir='data', batch_size=32, dataAugmentation=config.data_aug)
    wandb.run.name = (
        f"n_filters_{config.num_filters}_act_{config.activation_function}_"
        f"fof_{config.filter_organisation_factor}_dropout_{config.dropout}_"
        f"bn_{config.batch_norm}_data_aug_{config.data_aug}_hs_{config.hidden_size}_"
        f"n_blocks_{config.n_blocks}_num_epochs_{config.epochs}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomCNN(num_filters=config.num_filters,
                      hidden_size=config.hidden_size,
                      filter_organisation_factor=config.filter_organisation_factor,
                      dropout=config.dropout,
                      batch_norm=config.batch_norm,
                      n_blocks=config.n_blocks,
                      activation_function=config.activation_function).to(device)
    
    history = model.train_model(train_loader=train_loader,
                                val_loader=val_loader,
                                epochs=config.epochs,
                                learning_rate=0.001,
                                device=device)

    # Log metrics to wandb
    for epoch in range(len(history['train_loss'])):
        wandb.log({
            'train_loss': history['train_loss'][epoch],
            'val_loss': history['val_loss'][epoch],
            'train_acc': history['train_acc'][epoch],
            'val_acc': history['val_acc'][epoch],
            'epoch': epoch
        })
    torch.cuda.empty_cache()
    gc.collect()
    del model

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "activation_function": {"values": ["SiLU", "Mish", "GeLU"]},
        "batch_norm": {"values": [True, False]},
        "data_aug": {"values": [True, False]},
        "dropout": {"values": [0.2, 0.3, 0.4]},
        "epochs": {"values": [5, 10, 15]},
        "filter_organisation_factor": {"values": [0.5, 1, 1.5]},
        "hidden_size": {"values": [128, 256]},
        "num_filters": {"values": [16, 32, 64]},
        "n_blocks": {"values": [5]}
    }
}


project_name = "iNaturalist-CNN-Optimization"

# Create sweep
sweep_id = wandb.sweep(
    sweep_config,
    project="iNaturalist-CNN-Optimization",
)

# Run agent for N trials
wandb.agent(sweep_id=sweep_id, function=train, count=15)