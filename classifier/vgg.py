import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):

    def __init__(self, num_class=172):
        super(VGG19, self).__init__()

        self.vgg = models.vgg19(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9 * 9, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_class)
        )

    def forward(self, img):
        for idx, layer in self.vgg._modules.items():
            img = layer(img)
        img = img.view(img.size(0), -1)    
        img = self.classifier(img)
        return img
