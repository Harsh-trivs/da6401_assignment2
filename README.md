# Part A :

## Setup Instruction :

- Clone the repository  and move to part A code
    
    ```sql
    git clone https://github.com/Harsh-trivs/da6401_assignment1.git
    cd partA
    ```
    
- Create a virtual environment
    
    ```sql
    python -m venv venv
    ```
    
    Activate virtual environment
    
    ```powershell
    .\venv\Scripts\activate # Windows
    source venv/bin/activate # Mac/Linux
    ```
    
- Install dependencies
    
    ```powershell
    pip install -r requirements.txt
    ```
    

## Python files used for experiments :

### `model.py`

Defines the `CustomCNN` model class along with utility functions for data transformation and loading, which are shared across different experiments.

### `test_eval.py`

Handles training of a new model using the best-performing parameters obtained from hyperparameter sweeps, if a pretrained best model is not already available. It also logs the test accuracy and generates a prediction grid for evaluation.

### `wandbSweep.py`

Includes the code for running Weights & Biases (wandb) sweeps to perform hyperparameter tuning on the custom-built model.

# Part B :

## Setup Instruction :

- Clone the repository  and move to part B code
    
    ```sql
    git clone https://github.com/Harsh-trivs/da6401_assignment1.git
    cd partB
    ```
    
- Create a virtual environment
    
    ```sql
    python -m venv venv
    ```
    
    Activate virtual environment
    
    ```powershell
    .\venv\Scripts\activate # Windows
    source venv/bin/activate # Mac/Linux
    ```
    
- Install dependencies
    
    ```powershell
    pip install -r requirements.txt
    ```
    

## Python files used for experiments :

### fine_tuning.py

This script fine-tunes a **pretrained ResNet-50** model on the **iNaturalist dataset** (or any custom dataset structured in the same way) for a **10-class classification task and log metrics (train_loss, train_acc , val_acc , test_acc ,learning_rate ) on wandb .**

## Github Link :

 https://github.com/Harsh-trivs/da6401_assignment2

## Wandb Report :

https://api.wandb.ai/links/harshtrivs-indian-institute-of-technology-madras/fk4fk69x
