import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class GameNetV2(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        self.num_classes = 1
        self.model.fc = nn.Linear(2048, self.num_classes)
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

    def forward(self, x):
        return self.model(x)