# General purpose
import numpy as np

# Visualization
import plotly.graph_objects as go

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Miscelaneous
from tqdm.notebook import tqdm


def train_model(model, x_train, y_train, x_test, y_test, epochs=100, lr=0.01, criterion=nn.MSELoss()):
    """
    Train the model on the provided dataset.
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, betas=(0.9, 0.999))

    progress_bar = tqdm(range(epochs), desc="Progress")
    train_losses, test_losses = [], []
    weights_trajectory = []
    grad_norms = []
    for _ in progress_bar:
        model.train()
        
        optimizer.zero_grad()
        predictions = model(x_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        # Track training loss
        train_losses.append(loss.item())        

        # Evaluate on test set
        with torch.no_grad():
            model.eval()
            test_loss = criterion(model(x_test_tensor), y_test_tensor)
            # Track test loss
            test_losses.append(test_loss.item())

        # Track the weights evolution
        current_weights = [param.detach().cpu().numpy().copy() for param in model.parameters()]
        weights_trajectory.append(current_weights)

        # Track gradient norm
        grad_norm = 0
        for param in model.parameters():
            grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        grad_norms.append(grad_norm)

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    return train_losses, test_losses, weights_trajectory, grad_norms






def visualize_loss_landscape_3d_plotly(model, weights_trajectory, x, y, grid_size=50, radius=5):
    """
    Visualize the loss landscape in 3D by sampling over a grid in parameter space,
    with a user-defined radius around the current point.
    """
    # Copy the model to ensure the original model is not modified
    model = model.copy()

    # Ensure the data points are on the device
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    criterion = nn.MSELoss()

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
    current_loss = criterion(model(x_tensor), y_tensor).item()

    # Compute loss values
    losses = np.zeros_like(alpha_grid)
    for i in range(grid_size):
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
                predictions = model(x_tensor)
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
            predictions = model(x_tensor)
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