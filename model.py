import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    BACKBONE_NAME,
    DROPOUT_RATE,
    FREEZE_STAGES,
    HEAD_HIDDEN_DIMS,
    USE_IMAGENET_PRETRAINED,
)


class HeightPredictor(nn.Module):
    """Single-branch CNN regressor for patient height prediction."""

    def __init__(self, backbone_name=BACKBONE_NAME):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        self.feature_extractor, feature_dim = self._build_backbone(self.backbone_name)
        self._freeze_backbone_stages(FREEZE_STAGES)

        dims = [feature_dim] + list(HEAD_HIDDEN_DIMS)
        layers = [nn.Dropout(DROPOUT_RATE)]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-1], 1)]
        self.head = nn.Sequential(*layers)

    def _build_backbone(self, backbone_name):
        if backbone_name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAINED else None
            net = models.efficientnet_v2_s(weights=weights)
            net.classifier = nn.Identity()
            return net, 1280

        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAINED else None
            net = models.resnet18(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feature_dim

        if backbone_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAINED else None
            net = models.resnet34(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feature_dim

        raise ValueError(f'Unsupported BACKBONE_NAME: {backbone_name}')

    def _freeze_backbone_stages(self, num_stages):
        if num_stages <= 0:
            return

        if self.backbone_name == 'efficientnet_v2_s':
            for idx, block in enumerate(self.feature_extractor.features.children()):
                if idx < num_stages:
                    for p in block.parameters():
                        p.requires_grad = False
        elif self.backbone_name in {'resnet18', 'resnet34'}:
            blocks = [
                self.feature_extractor.conv1,
                self.feature_extractor.bn1,
                self.feature_extractor.layer1,
                self.feature_extractor.layer2,
                self.feature_extractor.layer3,
                self.feature_extractor.layer4,
            ]
            for idx, block in enumerate(blocks):
                if idx < num_stages:
                    for p in block.parameters():
                        p.requires_grad = False

    def forward(self, images):
        feats = self.feature_extractor(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return self.head(feats)


def create_model(device='cuda', backbone_name=BACKBONE_NAME, **kwargs):
    # kwargs kept for backward compatibility with older scripts.
    model = HeightPredictor(backbone_name=backbone_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model
