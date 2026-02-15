import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from config import (
    RESNET_FEATURE_DIM, METADATA_DIM, METADATA_HIDDEN_DIM,
    REGRESSOR_HIDDEN_DIM, DROPOUT_RATE
)


class RadImageNetHubHeightPredictor(nn.Module):
    """
    Height prediction model using RadImageNet pretrained ResNet50.

    Architecture:
    - ResNet50 backbone (pretrained on RadImageNet)
    - Metadata branch for pixel spacing
    - Regression head combining image and metadata features
    """

    def __init__(self, weights_path: str = None, freeze_backbone: bool = False):
        """
        Args:
            weights_path: Path to RadImageNet pretrained weights (.pt file)
            freeze_backbone: Whether to freeze ResNet backbone weights
        """
        super(RadImageNetHubHeightPredictor, self).__init__()

        # 1. Initialize ResNet50 backbone
        print("Initializing ResNet50 backbone...")
        self.backbone = models.resnet50(weights=None)

        # 2. Load RadImageNet weights if provided
        if weights_path is not None:
            self._load_pretrained_weights(weights_path)

        # 3. Remove original FC layer (we'll add our own regression head)
        self.backbone.fc = nn.Identity()

        # 4. Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

        # 5. Metadata branch for pixel spacing (2D -> 8D)
        self.meta_fc = nn.Sequential(
            nn.Linear(METADATA_DIM, METADATA_HIDDEN_DIM),
            nn.ReLU()
        )

        # 6. Regression head (ResNet features + metadata -> height prediction)
        self.regressor = nn.Sequential(
            nn.Linear(RESNET_FEATURE_DIM + METADATA_HIDDEN_DIM, REGRESSOR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(REGRESSOR_HIDDEN_DIM, 1)
        )

    def _load_pretrained_weights(self, weights_path: str):
        """Load RadImageNet pretrained weights"""
        weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found at {weights_path}. "
                "Please download RadImageNet weights or provide correct path."
            )

        print(f"Loading RadImageNet weights from: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location="cpu")

            # Remove 'module.' prefix if present (from DataParallel training)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v

            # Load weights (strict=False because we're replacing the FC layer)
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)

            print(f"✓ Weights loaded successfully")
            print(f"  Missing keys (expected for FC layer): {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained weights: {e}")

    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        print("Freezing ResNet50 backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        print("Unfreezing ResNet50 backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images, spacings):
        """
        Forward pass.

        Args:
            images: Tensor of shape (batch, channels, height, width)
            spacings: Tensor of shape (batch, 2) with pixel spacing metadata

        Returns:
            Tensor of shape (batch, 1) with height predictions
        """
        # Ensure 3-channel input
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Extract image features from ResNet
        img_feats = self.backbone(images)  # (batch, 2048)

        # Process metadata
        meta_feats = self.meta_fc(spacings)  # (batch, 8)

        # Combine features
        combined = torch.cat((img_feats, meta_feats), dim=1)  # (batch, 2056)

        # Predict height
        height_pred = self.regressor(combined)  # (batch, 1)

        return height_pred

    def get_num_params(self):
        """Get total number of parameters and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_model(weights_path: str = None, device: str = 'cuda') -> RadImageNetHubHeightPredictor:
    """
    Factory function to create and initialize the model.

    Args:
        weights_path: Path to RadImageNet weights
        device: Device to move model to ('cuda' or 'cpu')

    Returns:
        Initialized model on specified device
    """
    model = RadImageNetHubHeightPredictor(weights_path=weights_path)

    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print model info
    total_params, trainable_params = model.get_num_params()
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    model = RadImageNetHubHeightPredictor(weights_path=None)

    # Test forward pass
    dummy_images = torch.randn(2, 3, 256, 256)
    dummy_spacings = torch.randn(2, 2)

    output = model(dummy_images, dummy_spacings)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_images.shape}")
    print(f"  Spacing shape: {dummy_spacings.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Model test passed!")