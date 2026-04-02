import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    BACKBONE_FEATURE_DIM,
    BACKBONE_NAME,
    DROPOUT_RATE,
    METADATA_DIM,
    METADATA_HIDDEN_DIM,
    REGRESSOR_HIDDEN_DIM,
    USE_IMAGENET_PRETRAINED,
)


class HeightPredictor(nn.Module):
    """
    Height prediction model using EfficientNetV2 backbone + pixel-spacing metadata.

    init_mode semantics:
    -1: random initialization
     0: ImageNet pretrained
     1..4: ImageNet pretrained + progressively freeze earlier backbone stages
    """

    def __init__(self, dropout_rate: float = DROPOUT_RATE, init_mode: int = 0):
        super().__init__()

        use_pretrained = init_mode >= 0 and USE_IMAGENET_PRETRAINED
        self.backbone = self._build_backbone(BACKBONE_NAME, use_pretrained)

        if init_mode > 0:
            self._freeze_backbone_stages(init_mode)

        self.meta_fc = nn.Sequential(
            nn.Linear(METADATA_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, METADATA_HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(BACKBONE_FEATURE_DIM + METADATA_HIDDEN_DIM, REGRESSOR_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(REGRESSOR_HIDDEN_DIM, 1),
        )

    def _build_backbone(self, backbone_name: str, use_imagenet_pretrained: bool):
        if backbone_name != 'efficientnet_v2_s':
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Expected 'efficientnet_v2_s'.")

        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if use_imagenet_pretrained else None
        print(
            f"Initializing {backbone_name} backbone "
            f"({'ImageNet pretrained' if weights is not None else 'random init'})..."
        )

        backbone = models.efficientnet_v2_s(weights=weights)
        backbone.classifier = nn.Identity()
        return backbone

    def _freeze_backbone_stages(self, freeze_level: int):
        """Freeze progressively larger prefix of EfficientNet feature stages."""
        freeze_level = int(max(1, min(4, freeze_level)))

        # efficientnet_v2_s.features typically has 8 top-level blocks.
        total_feature_blocks = len(self.backbone.features)
        blocks_to_freeze = max(1, int(round((freeze_level / 4.0) * total_feature_blocks)))

        print(
            f"Freezing EfficientNetV2 backbone stages: "
            f"level={freeze_level}, blocks={blocks_to_freeze}/{total_feature_blocks}"
        )

        for stage_idx, stage in enumerate(self.backbone.features):
            if stage_idx < blocks_to_freeze:
                for param in stage.parameters():
                    param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing EfficientNetV2 backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images, spacings):
        # Ensure 3-channel input
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        img_feats = self.backbone(images)
        meta_feats = self.meta_fc(spacings)
        combined = torch.cat((img_feats, meta_feats), dim=1)
        return self.regressor(combined)

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_model(device: str = 'cuda', dropout_rate: float = DROPOUT_RATE, init_mode: int = 0) -> HeightPredictor:
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
    dummy_spacings = torch.randn(2, 2)

    output = model(dummy_images, dummy_spacings)
    print("\nTest forward pass:")
    print(f"  Input shape: {dummy_images.shape}")
    print(f"  Spacing shape: {dummy_spacings.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Model test passed!")
