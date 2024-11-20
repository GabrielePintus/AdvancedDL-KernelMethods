import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def load_mnist(seed=42):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Numpy arrays
    X_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
    X_test, y_test = test_dataset.data.numpy(), test_dataset.targets.numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)
    
    # Pandas DataFrames
    df_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    df_train['label'] = y_train
    df_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
    df_val['label'] = y_val
    df_test = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
    df_test['label'] = y_test
    
    return df_train, df_val, df_test


def build_data_loaders(X_train, y_train, X_val, y_val, batch_size=64, X_dtype=torch.float32, y_dtype=torch.float32):
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=X_dtype)
    y_train = torch.tensor(y_train, dtype=y_dtype)
    X_val = torch.tensor(X_val, dtype=X_dtype)
    y_val = torch.tensor(y_val, dtype=y_dtype)

    # Build torch dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    lr_scheduler,
    n_epochs,
    grad_clip,
    device
):
    train_losses = []
    val_losses = []
    lr_evol = []

    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 1e-3  # Minimum change to qualify as an improvement
    best_val_loss = float("inf")
    epochs_no_improve = 0

    progress_bar = tqdm(range(n_epochs), desc='Progress')
    for epoch in progress_bar:
        model.train()
        train_loss = 0.0
        lr_evol.append(optimizer.param_groups[0]['lr'])
        
        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update weights
            optimizer.step()
            
            # Update loss
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
        
        lr_scheduler.step(val_loss)
        # lr_scheduler.step()
        
        progress_bar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss})

        # Check if validation loss improved
        if epoch > n_epochs//10:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Early stopping condition
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
    
    return train_losses, val_losses, lr_evol



def compute_preds(model, data_loader, device):
    preds = []
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            y_pred = model(X)
            preds.append(y_pred.cpu().numpy())
    preds = np.vstack(preds)
    return preds.argmax(axis=1)

def compute_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).argmax(1)
            y_class = y.argmax(1)
            correct += (y_pred == y_class).sum().item()
            total += y.size(0)
    return correct / total


class Plots:

    @staticmethod
    def plot_2D(X, y):
        fig, ax = plt.subplots(1,1 , figsize=(6, 5))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax, hue=y, palette='viridis')
        ax.set_xlabel('First component')
        ax.set_ylabel('Second component')
        ax.set_title('First two PCA components')
        plt.show()

    def plot_3D(X, y):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
        ax.set_xlabel('First component')
        ax.set_ylabel('Second component')
        ax.set_zlabel('Third component')
        ax.set_title('First three PCA components')
        plt.show()

    @staticmethod
    def plot_explained_variance_ratio(x, y, ax, title):
        sns.barplot(x=x, y=y, ax=ax, color='skyblue')
        ax.set_title(title)
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Explained variance ratio')
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_ylim([5e-3, 1])
        ax.set_yticks([0.01, 0.05, 0.1, 0.5, 1])
        ax.set_yticklabels(['1%', '5%', '10%', '50%', '100%'])

    @staticmethod
    def loss_plot(train_losses, val_losses):
        print("Final training loss:", train_losses[-1])
        print("Final validation loss:", val_losses[-1])
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(train_losses, label='Train loss')
        ax.plot(val_losses, label='Val loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and validation loss')
        ax.legend()
        ax.grid(True)
        plt.show()

    @staticmethod
    def plot_3D_interactive(X, y):
        fig = go.Figure(data=[go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=y,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()


class Metrics:

    @staticmethod
    def harmonic_weighted_mean(x):
        harmonic_series = np.ones_like(x) / np.arange(1, len(x)+1)
        return np.sum(harmonic_series * x) / np.sum(harmonic_series)
