import torch.nn as nn
from torch.nn import init
from torchvision.models import resnet18

class resnet18_100(nn.Module):
    def __init__(self):
        super(resnet18_100, self).__init__()
        self.model = resnet18(pretrained = False)

        self.model.fc = nn.Linear(512, 100)
        init.kaiming_normal_(self.model.fc.weight.data)
        self.model.fc.bias.data.zero_()
    def forward(self,x):
        return self.model(x)
    

