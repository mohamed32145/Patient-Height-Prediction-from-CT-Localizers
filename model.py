import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    DROPOUT_RATE,
    METADATA_DIM,
    METADATA_HIDDEN_DIM,
    REGRESSOR_HIDDEN_DIM,
    USE_IMAGENET_PRETRAINED,
)

# --- NEW: We calculate this dynamically now ---
# 256 channels * 16 vertical zones = 4096
COMPRESSED_FEATURE_DIM = 4096


class HeightPredictor(nn.Module):
    """
    Height prediction model using EfficientNetV2 features + 1x1 Conv + Vertical Pooling.
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        # 1. Load the raw backbone
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAINED else None
        print(f"Initializing efficientnet_v2_s backbone...")
        full_backbone = models.efficientnet_v2_s(weights=weights)

        # 2. Extract ONLY the feature layers
        # (This leaves behind the useless 1x1 Global Average Pool and Classifier)
        self.feature_extractor = full_backbone.features

        if freeze_backbone:
            self._freeze_backbone()

        # 3. The Channel Compressor (1x1 Convolution)
        # Squeezes 1280 channels down to 256 to prevent parameter explosion
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)  # SiLU is the native activation function for EfficientNet
        )

        # 4. The Vertical Ruler
        # Averages width to 1, but preserves 16 height zones
        self.vertical_pool = nn.AdaptiveAvgPool2d((16, 1))

        # 5. Metadata Branch
        self.meta_fc = nn.Sequential(
            nn.Linear(METADATA_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(32, METADATA_HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

        # 6. Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(COMPRESSED_FEATURE_DIM + METADATA_HIDDEN_DIM, REGRESSOR_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(REGRESSOR_HIDDEN_DIM, 1),
        )

    def _freeze_backbone(self):
        print("Freezing EfficientNetV2 features...")
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing EfficientNetV2 features...")
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, images, spacings):
        # Ensure 3-channel input
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # --- THE SPATIAL PIPELINE ---

        # Step 1: Extract features (Output shape: Batch, 1280 channels, Height/32, Width/32)
        x = self.feature_extractor(images)

        # Step 2: Compress channels (Output shape: Batch, 256 channels, Height/32, Width/32)
        x = self.channel_compressor(x)

        # Step 3: Vertical Pool (Output shape: Batch, 256 channels, 16 vertical zones, 1 horizontal zone)
        x = self.vertical_pool(x)

        # Step 4: Flatten for the linear layer (Output shape: Batch, 4096)
        img_feats = torch.flatten(x, 1)

        # --- THE METADATA PIPELINE ---
        meta_feats = self.meta_fc(spacings)

        # --- COMBINE & PREDICT ---
        combined = torch.cat((img_feats, meta_feats), dim=1)
        return self.regressor(combined)

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_model(device: str = 'cuda') -> HeightPredictor:
    model = HeightPredictor(freeze_backbone=False)

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
    dummy_spacings = torch.randn(2, 2)

    output = model(dummy_images, dummy_spacings)
    print("\nTest forward pass:")
    print(f"  Input shape: {dummy_images.shape}")
    print(f"  Spacing shape: {dummy_spacings.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Model test passed!")