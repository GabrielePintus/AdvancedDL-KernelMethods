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
            # progress_bar.set_description(f"Train Loss: {losses_train[-1]:.1e}, Test Loss: {losses_test[-1]:.1e}")
            progress_bar.set_postfix({'Train Loss': losses_train[-1], 'Test Loss': losses_test[-1]})
                
            progress_bar.update(test_every)
                
    return losses_train, losses_test
        
        
def interpolate_test_losses(losses_test, n_steps, test_every):
    interpolated_test_losses = np.full(n_steps, np.nan)
    interpolated_test_losses[::test_every] = losses_test
    interpolated_test_losses = pd.Series(interpolated_test_losses).interpolate()
    return interpolated_test_losses
    
