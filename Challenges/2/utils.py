import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from tqdm.notebook import tqdm



def generate_test_dataset(n, teacher, device):
    X = np.random.uniform(0, 2, (n, 100))
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = teacher(X)
    dataset = TensorDataset(X, y)
    return dataset


def sample_batch(teacher, batch_size, device):
    with torch.no_grad():
        X = np.random.uniform(0, 2, (batch_size, 100))
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = teacher(X)
    return X, y


def print_nparams(model):
    nparams = sum(p.numel() for p in model.parameters())
    obj_name = model.__class__.__name__
    print(f"Number of parameters for {obj_name}: {nparams}")
    
    
def train(
    student,
    teacher, 
    test_loader,
    lr,    
    n_steps=1000,
    test_every=100,
    device = torch.device('cpu')
):
    losses_train, losses_test = [], []
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    progress_bar = tqdm(range(n_steps), desc='Training', leave=True, position=0)
    
    for step in progress_bar:
        student.train()
        
        # Sample a batch of data
        X, y = sample_batch(teacher, batch_size=128, device=device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = student(X)
        
        # Compute loss
        loss = criterion(y_pred, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        # Optimize
        optimizer.step()
        
        # Compute train loss
        losses_train.append(torch.mean(loss).item())        
        
        # Compute test loss
        if step % test_every == 0:
            student.eval()
            loss = 0
            for X, y in test_loader:
                y_pred = student(X)
                loss += criterion(y_pred, y).item()
            loss /= len(test_loader)
            losses_test.append(loss)
            
            # Update progress bar logging train and test loss
            progress_bar.set_postfix({'Train Loss': np.mean(losses_train[-3:]), 'Test Loss': np.mean(losses_test[-3:])})                
            progress_bar.update(test_every)
                
    return losses_train, losses_test


def train_resnet(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    n_epochs=10,
    test_every=1,
    device = torch.device('cpu')
):
    losses_train, losses_test = [], []    
    progress_bar = tqdm(range(n_epochs), desc='Training', leave=True, position=0)
    
    for epoch in progress_bar:
        model.train()
        tmp_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X)
            
            # Compute loss
            loss = criterion(y_pred, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimize
            optimizer.step()
            
            # Compute train loss
            tmp_loss += loss.item()
        losses_train.append(tmp_loss / len(train_loader))
            
        # Compute test loss
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            tmp_loss += loss.item()
        losses_test.append(tmp_loss / len(test_loader))
            
        # Update progress bar logging train and test loss
        progress_bar.set_postfix({'Train Loss': np.mean(losses_train[-3:]), 'Test Loss': np.mean(losses_test[-3:])})                
            
    return losses_train, losses_test


        
def interpolate_test_losses(losses_test, n_steps, test_every):
    interpolated_test_losses = np.full(n_steps, np.nan)
    interpolated_test_losses[::test_every] = losses_test
    interpolated_test_losses = pd.Series(interpolated_test_losses).interpolate()
    return interpolated_test_losses
    

# Second exercise
from sympy import symbols, lambdify
import random

# Define symbolic variables
x1, x2, x3, x4, x5, x6 = symbols("x1 x2 x3 x4 x5 x6")

# Define the hierarchical Bell polynomial B6
B6 = (
    x1**6
    + 15 * x2 * x1**4
    + 20 * x3 * x1**3
    + 45 * x2**2 * x1**2
    + 15 * x2**3
    + 60 * x3 * x2 * x1
    + 15 * x4 * x1**2
    + 10 * x3**2
    + 15 * x4 * x2
    + 6 * x5 * x1
    + x6
)
B6_scrambled = (
    x6
    + 6 * x5 * x1
    + 15 * x4 * x2
    + 10 * x3**2
    + 15 * x4 * x1**2
    + 60 * x3 * x2 * x1
    + 15 * x2**3
    + 45 * x2**2 * x1**2
    + 20 * x3 * x1**3
    + 15 * x2 * x1**4
    + x1**6
)

# Create a callable function for numerical evaluation
bell6 = lambdify([x1, x2, x3, x4, x5, x6], B6, "numpy")
bell6_scrambled = lambdify([x1, x2, x3, x4, x5, x6], B6_scrambled, "numpy")

