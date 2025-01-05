import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm


def train(model, train_loader, test_loader, optimizer, num_epochs, device):
    train_losses, test_losses = [], []
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                test_loss += F.mse_loss(output, target).item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss)
    return train_losses, test_losses