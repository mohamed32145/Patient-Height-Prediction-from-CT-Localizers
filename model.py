import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    DROPOUT_RATE,
    REGRESSOR_HIDDEN_DIM,
    USE_IMAGENET_PRETRAINED,
)

# 256 channels * 16 vertical zones = 4096
COMPRESSED_FEATURE_DIM = 4096


class HeightPredictor(nn.Module):
    """
    Height prediction model using EfficientNetV2 features + 1x1 Conv + Vertical Pooling.

    Image-only: the model predicts height purely from the image. The pixel-spacing
    metadata branch has been removed; spacing is still used for image preprocessing
    (resampling to isotropic spacing) but is no longer fed to the network.
    """

    # UPDATED: Added dropout_rate and init_mode arguments
    def __init__(self, dropout_rate=None, init_mode=0):
        super().__init__()

        # Use provided dropout or fallback to config
        self.dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE

        # 1. Load the raw backbone
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAINED else None
        print(f"Initializing efficientnet_v2_s backbone...")
        full_backbone = models.efficientnet_v2_s(weights=weights)

        # 2. Extract ONLY the feature layers
        self.feature_extractor = full_backbone.features

        # Apply partial freezing based on init_mode
        self._apply_init_mode(init_mode)

        # 3. The Channel Compressor (1x1 Convolution)
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        # 4. The Vertical Ruler
        self.vertical_pool = nn.AdaptiveAvgPool2d((16, 1))

        # 5. Regression Head (image features only)
        self.regressor = nn.Sequential(
            nn.Linear(COMPRESSED_FEATURE_DIM, REGRESSOR_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),  # Using dynamic dropout
            nn.Linear(REGRESSOR_HIDDEN_DIM, 1),
        )

    def _apply_init_mode(self, init_mode):
        """Freezes stages of the backbone based on init_mode integer."""
        if init_mode <= 0:
            print("init_mode=0: No layers frozen.")
            return

        print(f"Applying init_mode={init_mode} (Freezing first {init_mode} stages of backbone)...")
        # EfficientNetV2 features are a Sequential block. We freeze the first 'init_mode' blocks.
        for i, child in enumerate(self.feature_extractor.children()):
            if i < init_mode:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_backbone(self):
        print("Unfreezing EfficientNetV2 features...")
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        x = self.feature_extractor(images)
        x = self.channel_compressor(x)
        x = self.vertical_pool(x)
        img_feats = torch.flatten(x, 1)

        return self.regressor(img_feats)

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# UPDATED: Accept dropout_rate and init_mode
def create_model(device: str = 'cuda', dropout_rate: float = None, init_mode: int = 0) -> HeightPredictor:
    model = HeightPredictor(dropout_rate=dropout_rate, init_mode=init_mode)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params, trainable_params = model.get_num_params()
    print("\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    return model


if __name__ == "__main__":
    print("Testing model creation...")
    model = HeightPredictor()

    dummy_images = torch.randn(2, 3, 384, 384)

    output = model(dummy_images)
    print("\nTest forward pass:")
    print(f"  Input shape: {dummy_images.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Model test passed!")