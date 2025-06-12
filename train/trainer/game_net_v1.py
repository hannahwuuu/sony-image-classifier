import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class GameNetV1(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.num_classes = 1
        self.model.classifier[3] = nn.Linear(1024, self.num_classes)
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

    def forward(self, x):
        return self.model(x)