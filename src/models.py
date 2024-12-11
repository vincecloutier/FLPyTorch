from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

# class CNNFashion(nn.Module):
#     def __init__(self, args):
#         super(CNNFashion, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1,32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 64 * 7 * 7)
#         x = self.classifier(x)
#         return x


# resnet for FMNIST
# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_planes)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)  
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != out_planes:
#             self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_planes))

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class CNNFashion(nn.Module):
#     def __init__(self, args, num_classes=10):
#         super(CNNFashion, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(16, 16, stride=1)
#         self.layer2 = self._make_layer(16, 32, stride=2)
#         self.layer3 = self._make_layer(32, 64, stride=2)
#         self.linear = nn.Linear(64, num_classes)

#     def _make_layer(self, in_planes, out_planes, stride):
#         return nn.Sequential(BasicBlock(in_planes, out_planes, stride=stride))

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x))) 
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)  
#         out = F.avg_pool2d(out, out.size(2))
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


class CNNFashion(nn.Module):
    def __init__(self, args, in_channels = 3, num_classes = 10):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        # return F.log_softmax(out, dim=1)
        return out # for cross entropy loss


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return x # for cross entropy loss

#resnet9 for cifar
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, args, in_channels = 3, num_classes = 10):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        # return F.log_softmax(out, dim=1)
        return out # for cross entropy loss


# resnet for imagenet
class ImageNetModel(nn.Module):
    def __init__(self, args):
        super(ImageNetModel, self).__init__()
        # load pretrained ResNet50
        self.model = resnet50(pretrained=True)
        
        # freeze all layers except the last two
        for param in list(self.model.parameters())[:-2]:
            param.requires_grad = False
            
        # replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1000)

    def forward(self, x):
        return self.model(x)