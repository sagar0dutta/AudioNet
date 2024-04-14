import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

resnet_dict = {'ResNet18': resnet18, 'ResNet34': resnet34, 'ResNet50': resnet50}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet34"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=False)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = nn.Identity()
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)

        mean_val = torch.mean(y, dim=1, keepdim=True)  
        y = torch.where(y >= mean_val, torch.ones_like(y), -torch.ones_like(y))

        return y, x
