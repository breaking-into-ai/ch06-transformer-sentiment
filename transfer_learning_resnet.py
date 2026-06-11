"""Chapter 6 - transfer learning: adapt a pre-trained ResNet to movie poster genres."""
from torch import nn
from torchvision import models

resnet = models.resnet18(weights="IMAGENET1K_V1")  # pre-trained on 1M+ images
for param in resnet.parameters():
    param.requires_grad = False                     # freeze the backbone
resnet.fc = nn.Linear(resnet.fc.in_features, 10)    # new head: 10 genres
# Now only the new layer trains - fast and effective even with small datasets.
print(resnet.fc)
