# General purpose
import numpy as np

# Visualization
import plotly.graph_objects as go
from plotly.offline import plot

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Metrics
from sklearn.metrics import balanced_accuracy_score

# Miscelaneous
from tqdm.notebook import tqdm
from copy import deepcopy
from joblib import Parallel, delayed



def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, n_epochs, device):
    progress_bar = tqdm(range(n_epochs), desc="Epochs")
    train_losses, test_losses = [], []
    bacc_train, bacc_test = [], []
    weights_trajectory = []
    grad_norms = []

    for epoch in progress_bar:
        model.train()

        train_loss = 0
        y_true_train, y_pred_train = [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            train_loss += loss.item()

            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(y_hat.argmax(dim=1).cpu().numpy())
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        bacc_train.append(balanced_accuracy_score(y_true_train, y_pred_train))

        model.eval()
        test_loss = 0
        y_true_test, y_pred_test = [], []
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            test_loss += loss.item()

            y_true_test.extend(y.cpu().numpy())
            y_pred_test.extend(y_hat.argmax(dim=1).cpu().numpy())
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        bacc_test.append(balanced_accuracy_score(y_true_test, y_pred_test))
        
        # Save the weights
        weights = [p.detach().cpu().numpy().copy() for p in model.parameters()]
        weights_trajectory.append(weights)
        # Compute the gradient norm
        grad_norm = np.linalg.norm(np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters()]))
        grad_norms.append(grad_norm)

        progress_bar.set_postfix({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_bacc": bacc_train[-1],
            "test_bacc": bacc_test[-1]
        })

    return (train_losses, test_losses), (bacc_train, bacc_test), (weights_trajectory, grad_norms)


def visualize_loss_landscape_3d_plotly(model, weights_trajectory, X_tensor, y_tensor, grid_size=50, radius=5):
    """
    Visualize the loss landscape in 3D by sampling over a grid in parameter space,
    with a user-defined radius around the current point.
    """
    # Copy the model to ensure the original model is not modified
    model = deepcopy(model)

    # Ensure the data points are on the device
    criterion = nn.CrossEntropyLoss()

    # Ensure the model is in evaluation mode
    model.eval()

    # Collect parameters
    params = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    num_params = len(params)

    # Generate two random orthogonal directions
    direction1 = np.random.randn(num_params)
    direction2 = np.random.randn(num_params)

    # # Extract an orthogonal basis using Gram-Schmidt
    # direction2 -= direction1 * np.dot(direction2, direction1) / np.linalg.norm(direction1)**2  # Orthogonalize
    # direction1 /= np.linalg.norm(direction1)
    # direction2 /= np.linalg.norm(direction2)

    # # Take the canonical basis vectors
    # direction1 = np.eye(num_params)[0]
    # direction2 = np.eye(num_params)[1]

    # Choose the direction as the main axes of variation in the weights
    pca = PCA(n_components=2)
    pca.fit(np.array([ np.concatenate([ w.reshape(-1) for w in weights ]) for weights in weights_trajectory ]))
    direction1, direction2 = pca.components_

    # Generate grid within the given radius
    alphas = np.linspace(-radius, radius, grid_size)
    betas = np.linspace(-radius, radius, grid_size)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)

    # Compute the loss value of the current model
    current_loss = criterion(model(X_tensor), y_tensor).item()

    # Compute loss values
    losses = np.zeros_like(alpha_grid)
    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            perturbed_params = params + alpha_grid[i, j] * direction1 + beta_grid[i, j] * direction2
            start = 0
            with torch.no_grad():
                for param in model.parameters():
                    numel = param.numel()
                    param.copy_(
                        torch.tensor(perturbed_params[start:start + numel]).reshape(param.shape)
                    )
                    start += numel
                predictions = model(X_tensor)
                loss = criterion(predictions, y_tensor).item()
                losses[i, j] = loss

    # Project weight trajectory onto the two directions
    trajectory_points = []
    for weights in weights_trajectory:
        flat_weights = np.concatenate([w.flatten() for w in weights])
        alpha = np.dot(flat_weights - params, direction1)
        beta = np.dot(flat_weights - params, direction2)
        trajectory_points.append((alpha, beta))

    trajectory_points = np.array(trajectory_points)
    trajectory_alphas = trajectory_points[:, 0]
    trajectory_betas = trajectory_points[:, 1]
    trajectory_losses = []
    # Compute the loss values for the trajectory points
    for weights in weights_trajectory:
        start = 0
        with torch.no_grad():
            for param, w in zip(model.parameters(), weights):
                numel = param.numel()
                param.copy_(torch.tensor(w).reshape(param.shape))
                start += numel
            predictions = model(X_tensor)
            trajectory_losses.append(criterion(predictions, y_tensor).item())


    # Plot the 3D loss landscape using Plotly
    fig = go.Figure(data=[go.Surface(z=losses, x=alpha_grid, y=beta_grid, colorscale='Viridis')])

    # Add the current point
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[current_loss],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=['Current Point'],
        textposition='top center',
        name='Current Point'
    ))

    # Add the trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory_alphas,
        y=trajectory_betas,
        z=trajectory_losses,
        mode='lines+markers',
        marker=dict(size=5, color='blue'),
        line=dict(color='blue', width=2),
        name='Weight Trajectory'
    ))

    # Update layout for better visuals
    fig.update_layout(
        title='3D Loss Landscape',
        scene=dict(
            xaxis_title='Direction 1',
            yaxis_title='Direction 2',
            zaxis_title='Loss',
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Show the plot
    fig.show()
    




def visualize_loss_landscape_3d_plotly_browser(model, weights_trajectory, X_tensor, y_tensor, grid_size=50, radius=5, name="model", traj_points=0):
    """
    Visualize the loss landscape in 3D by sampling over a grid in parameter space,
    with a user-defined radius around the current point.
    """
    # Copy the model to ensure the original model is not modified
    model = deepcopy(model)

    # Ensure the data points are on the device
    criterion = nn.CrossEntropyLoss()

    # Ensure the model is in evaluation mode
    model.eval()

    # Collect parameters
    params = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    num_params = len(params)

    # Generate two random orthogonal directions
    direction1 = np.random.randn(num_params)
    direction2 = np.random.randn(num_params)

    # Choose the direction as the main axes of variation in the weights
    pca = PCA(n_components=2)
    pca.fit(np.array([ np.concatenate([ w.reshape(-1) for w in weights ]) for weights in weights_trajectory ]))
    direction1, direction2 = pca.components_

    # Generate grid within the given radius
    alphas = np.linspace(-radius, radius, grid_size)
    betas = np.linspace(-radius, radius, grid_size)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)

    # Compute the loss value of the current model
    current_loss = criterion(model(X_tensor), y_tensor).item()

    # Compute loss values
    def compute_loss(i, j):
        perturbed_params = params + alpha_grid[i, j] * direction1 + beta_grid[i, j] * direction2
        start = 0
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()
                param.copy_(
                    torch.tensor(perturbed_params[start:start + numel]).reshape(param.shape)
                )
                start += numel
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor).item()
        return i, j, loss
    # losses = np.zeros_like(alpha_grid)
    # for i in tqdm(range(grid_size), leave=False):
    #     for j in range(grid_size):
    #         perturbed_params = params + alpha_grid[i, j] * direction1 + beta_grid[i, j] * direction2
    #         start = 0
    #         with torch.no_grad():
    #             for param in model.parameters():
    #                 numel = param.numel()
    #                 param.copy_(
    #                     torch.tensor(perturbed_params[start:start + numel]).reshape(param.shape)
    #                 )
    #                 start += numel
    #             predictions = model(X_tensor)
    #             loss = criterion(predictions, y_tensor).item()
    #             losses[i, j] = loss
    # Parallel computation of losses
    results = Parallel(n_jobs=-1)(delayed(compute_loss)(i, j) for i in range(grid_size) for j in range(grid_size))

    # Assemble results into a 2D losses array
    losses = np.zeros((grid_size, grid_size))
    for i, j, loss in results:
        losses[i, j] = loss

    # Project weight trajectory onto the two directions
    trajectory_points = []
    for weights in weights_trajectory:
        flat_weights = np.concatenate([w.flatten() for w in weights])
        alpha = np.dot(flat_weights - params, direction1)
        beta = np.dot(flat_weights - params, direction2)
        trajectory_points.append((alpha, beta))

    trajectory_points = np.array(trajectory_points)
    trajectory_alphas = trajectory_points[:, 0]
    trajectory_betas = trajectory_points[:, 1]
    trajectory_losses = []
    # Compute the loss values for the trajectory points
    for weights in weights_trajectory:
        start = 0
        with torch.no_grad():
            for param, w in zip(model.parameters(), weights):
                numel = param.numel()
                param.copy_(torch.tensor(w).reshape(param.shape))
                start += numel
            predictions = model(X_tensor)
            trajectory_losses.append(criterion(predictions, y_tensor).item())


    # Plot the 3D loss landscape using Plotly
    # Create the figure
    fig = go.Figure()

    # Add the surface plot
    fig.add_trace(go.Surface(z=losses, x=alpha_grid, y=beta_grid, colorscale='Viridis'))

    # Add the current point
    fig.add_trace(go.Scatter3d(
        x=[0],  # Replace with the actual x-coordinate of the current point
        y=[0],  # Replace with the actual y-coordinate of the current point
        z=[current_loss],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=['Current Point'],
        textposition='top center',
        name='Current Point'
    ))

    # Add the trajectory - plot last `traj_points` points
    if traj_points > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory_alphas[-traj_points:],
            y=trajectory_betas[-traj_points:],
            z=trajectory_losses[-traj_points:],
            mode='lines+markers',
            marker=dict(size=5, color='blue'),
            line=dict(color='blue', width=2),
            name='Weight Trajectory'
        ))

    # Update layout
    fig.update_layout(
        title='3D Loss Landscape',
        scene=dict(
            xaxis_title='Direction 1',
            yaxis_title='Direction 2',
            zaxis_title='Loss',
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Render in the browser
    filename = f'3d_loss_landscape_{name}.html'
    plot(fig, filename=filename, auto_open=True)
    