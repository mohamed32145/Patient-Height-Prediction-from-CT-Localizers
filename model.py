import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import config

class XRVHeightRegressor(nn.Module):
    def __init__(self, spacing_dim=2):
        super().__init__()
        print("Loading TorchXRayVision model...")
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").to(config.DEVICE)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = 1024
        self.spacing_mlp = nn.Sequential(
            nn.BatchNorm1d(spacing_dim),
            nn.Linear(spacing_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, spacing):
        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        s = self.spacing_mlp(spacing)
        combined = torch.cat([feats, s], dim=1)
        out = self.head(combined)
        return out