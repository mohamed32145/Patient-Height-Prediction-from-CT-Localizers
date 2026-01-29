import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import config
import random


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_mse, total_mae, total_n = 0.0, 0.0, 0

    for img_t, spacing_t, y_t in loader:
        img_t = img_t.to(config.DEVICE)
        spacing_t = spacing_t.to(config.DEVICE)
        y_t = y_t.to(config.DEVICE)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(img_t, spacing_t)
        loss = F.mse_loss(pred, y_t)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = img_t.size(0)
        total_mse += loss.item() * bs
        total_mae += (pred - y_t).abs().sum().item()
        total_n += bs

    return total_mse / total_n, total_mae / total_n


def compute_gradcam(model, input_image, spacing, target_layer):
    model.eval()
    activations_list, gradients_list = [], []

    def forward_hook(module, input, output):
        activations_list.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients_list.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    input_image.requires_grad_(True)
    output = model(input_image, spacing)
    model.zero_grad()
    output.backward(torch.ones_like(output), retain_graph=True)

    activations_np = activations_list[0].cpu().data.numpy()
    gradients_np = gradients_list[0].cpu().data.numpy()

    forward_handle.remove()
    backward_handle.remove()

    weights = np.mean(gradients_np, axis=(2, 3))[0]
    grad_cam = activations_np[0] * weights[:, None, None]
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = np.sum(grad_cam, axis=0)
    grad_cam = grad_cam - np.min(grad_cam)
    if np.max(grad_cam) != 0:
        grad_cam = grad_cam / np.max(grad_cam)
    grad_cam = cv2.resize(grad_cam, (config.IMG_SIZE, config.IMG_SIZE))
    return grad_cam


def visualize_sample(model, dataset, df):
    idx_to_visualize = 0
    sample_img_t, sample_spacing_t, sample_target_t = dataset[idx_to_visualize]
    sample_img_t = sample_img_t.unsqueeze(0).to(config.DEVICE)
    sample_spacing_t = sample_spacing_t.unsqueeze(0).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        pred_height = model(sample_img_t, sample_spacing_t).item()

    target_layer = model.backbone.features.norm5
    heatmap = compute_gradcam(model, sample_img_t, sample_spacing_t, target_layer)
    original_image_np = (sample_img_t.squeeze().cpu().detach().numpy() + 1024) / 2048

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np, cmap='gray')
    plt.title(f'True: {sample_target_t.item():.2f}cm, Pred: {pred_height:.2f}cm')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_image_np, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()

    print(f"Viz Patient: {df.iloc[idx_to_visualize]['Patient_ID']}")


def visualize_dataset_samples(dataset, num_samples=5, title_prefix="Sample"):
    """
    Picks random samples from the dataset and visualizes them.
    Handles the un-normalization from XRV range [-1024, 1024] back to [0, 1].
    """
    plt.figure(figsize=(15, 5))

    # Pick random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        # 1. Get the data from the dataset
        img_tensor, spacing_tensor, height_tensor = dataset[idx]

        # 2. Convert to Numpy
        # Shape is [1, 224, 224], squeeze to [224, 224]
        img_np = img_tensor.squeeze().cpu().numpy()

        # 3. Un-normalize for visualization
        # The dataset maps [0,1] -> [-1024, 1024]. We reverse this.
        # Formula: x_view = (x_xrv + 1024) / 2048
        img_view = (img_np + 1024) / 2048

        # Clip just in case floating point math went slightly out of bounds
        img_view = np.clip(img_view, 0, 1)

        # 4. Plot
        ax = plt.subplot(1, num_samples, i + 1)
        ax.imshow(img_view, cmap='gray')

        # Extract spacing for title
        sx, sy = spacing_tensor[0].item(), spacing_tensor[1].item()
        h_cm = height_tensor.item()

        ax.set_title(f"H: {h_cm:.1f}cm\nSpace: {sx:.2f}x{sy:.2f}", fontsize=10)
        ax.axis('off')

    plt.suptitle(f"{title_prefix} (Bone Window + Resize + Padding)", fontsize=14)
    plt.tight_layout()
    plt.show()