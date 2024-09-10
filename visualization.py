from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_loss_curve(loss_list, title):
    """
    Plot the loss curve across all epochs, highlighting the point with the minimum loss.

    Parameters:
    loss_list (list of float): A list containing the loss values for all epochs.
    title (str): The title of the chart.
    """
    epochs = range(1, len(loss_list) + 1)
    min_loss_epoch = loss_list.index(min(loss_list)) + 1
    min_loss_value = min(loss_list)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_list, label='Loss', color='blue', marker='o')
    plt.axvline(x=min_loss_epoch, color='red', linestyle='--',
                label=f'Min Loss Epoch: {min_loss_epoch}')
    plt.axhline(y=min_loss_value, color='green', linestyle='--',
                label=f'Min Loss Value: {min_loss_value:.4f}')
    plt.scatter(min_loss_epoch, min_loss_value, color='red', s=100, zorder=5)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_true, y_score, title):
    """
    Plot the ROC curve.

    Parameters:
    y_true (array-like): True labels.
    y_score (array-like): Predicted scores.
    title (str): The title of the chart.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title}')
    plt.legend(loc='lower right')
    plt.show()


def t_sne(title, model, dataset, dataset_name):
    """
    Performs t-SNE dimensionality reduction and visualizes the embedding results.

    Parameters:
    title (str): Title of the plot.
    model (torch.nn.Module): The model used to obtain embeddings.
    dataset (torch.utils.data.Dataset): The dataset for generating embeddings.
    dataset_name (str): The name of the dataset to choose color scheme.
    """
    # Define color scheme based on dataset
    if dataset_name == 'ABIDE':
        color_list = ['#FF8C00', '#1E90FF']
    else:
        color_list = ['#CD25C7', '#32CD32']

    # DataLoader to iterate through the dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embs = []
    colors = []

    # Set device for model inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch in loader:
        with torch.no_grad():
            batch.to(device)
            emb, _ = model(batch)
            embs.append(emb.detach().cpu().numpy())
            colors += [color_list[int(y.cpu())] for y in batch.y]

    embs = np.array(embs)
    embs = np.reshape(embs, (embs.shape[0], -1))

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=2024)
    embedding_2d = tsne.fit_transform(embs)

    # Visualize t-SNE results
    plt.figure(figsize=(10, 8))
    for color in np.unique(colors):
        indices = np.array(colors) == color
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1], color=color, label=color)

    plt.title(f't-SNE {title}')
    plt.legend()
    plt.show()


# Grad-CAM algorithm from https://doi.org/10.48550/arXiv.1610.02391
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        def forward_hook(module, input, output):
            self.activation = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        embedding, logits = self.model(input_tensor)
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0][target_class] = 1
        logits.backward(gradient=one_hot_output, retain_graph=True)

        gradients = self.gradient.cpu().data.numpy()[0]
        activations = self.activation.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=0)
        weights = weights.reshape(-1, 1)
        cam = np.array(np.dot(activations, weights), dtype=np.float32)

        cam = np.maximum(cam, 0)    # ReLU
        return cam



def grad_cam(model, dataset, n=10):
    """
    Computes and prints the top `n` indices with highest values for negative and positive classes
    using Grad-CAM.

    Parameters:
    model (torch.nn.Module): The model to use for Grad-CAM.
    dataset (torch.utils.data.Dataset): The dataset for generating Grad-CAM.
    n (int): Number of top indices to return (default is 10).
    """
    # Set device and model to evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Initialize Grad-CAM for the last convolutional layer
    target_layer = model.convs[-1]
    grad_cam = GradCAM(model, target_layer)

    # DataLoader to iterate through the dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    negative_cam = []
    positive_cam = []

    for batch in loader:
        batch = batch.to(device)
        target_class = batch.y.long().item()
        cam = grad_cam.generate_cam(batch, target_class)

        # Append CAM based on class label
        if target_class == 0:
            negative_cam.append(cam)
        else:
            positive_cam.append(cam)

    # Convert lists to NumPy arrays and compute mean across samples
    negative_cam = np.mean(np.array(negative_cam), axis=0)
    positive_cam = np.mean(np.array(positive_cam), axis=0)

    # Get top `n` indices for negative and positive CAMs
    negative_cam_indices = np.argsort(negative_cam.flatten())[::-1][:n]
    positive_cam_indices = np.argsort(positive_cam.flatten())[::-1][:n]

    # Print top `n` indices and their values for negative CAM
    print(f'Top {n} negative_cam_indices and values:')
    negative_output = ', '.join([f"{idx}:{negative_cam.flatten()[idx]}" for idx in negative_cam_indices])
    print(negative_output)

    # Print top `n` indices and their values for positive CAM
    print(f'Top {n} positive_cam_indices and values:')
    positive_output = ', '.join([f"{idx}:{positive_cam.flatten()[idx]}" for idx in positive_cam_indices])
    print(positive_output)

