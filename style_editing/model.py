import torch as th
from torchvision.models import resnet50
import torch.nn.functional as F


class Model(th.nn.Module):

    def __init__(self, num_outputs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = num_outputs

        resnet = resnet50(pretrained=True)
        self.features = th.nn.Sequential(*list(resnet.children())[:-1])
        # for param in self.features.parameters():
        #     param.requires_grad = False
        self.fc1 = th.nn.Linear(2048, self.num_outputs)
        self.bn1 = th.nn.BatchNorm2d(2048)

    def forward(self, x):
        x = self.features(x)
        x = self.bn1(x)
        x = th.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        return x

