from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
import torch

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


class MK_Block(nn.Module):
    def __init__(self, in_channels):
        super(MK_Block, self).__init__()
        
        # Define convolutional layers with different kernel sizes
        self.conv3 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # Reduced channels
        self.conv5 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, 8, kernel_size=7, padding=3)
        
        self.bn3 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn7 = nn.BatchNorm2d(8)
        
        # After first set of conv layers
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.bn5_2 = nn.BatchNorm2d(16)
        
        self.conv5_main = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.conv7_main = nn.Conv2d(8, 12, kernel_size=7, padding=3)
        self.bn5_main = nn.BatchNorm2d(24)
        self.bn7_main = nn.BatchNorm2d(12)
        
        self.conv3_final = nn.Conv2d(24, 36, kernel_size=3, padding=1)
        self.bn3_final = nn.BatchNorm2d(36)
        
        # Calculate the number of input channels for conv1x1
        # Original: in_channels + 32 + 16 + 8 + 24 + 12 + 36
        # Simplified channels after reductions: in_channels + 32 + 16 + 8 + 24 + 12 + 36 = in_channels + 128
        self.conv1x1 = nn.Conv2d(in_channels + 32 + 16 + 8 + 24 + 12 + 36, 24, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # First set of convolutions
        out_1_3 = F.relu(self.bn3(self.conv3(x)))  # [batch,32,H,W]
        out_1_5 = F.relu(self.bn5(self.conv5(x)))  # [batch,16,H,W]
        out_1_7 = F.relu(self.bn7(self.conv7(x)))  # [batch,8,H,W]
        
        # Concatenations after first set
        out_1_3_5 = torch.cat([x, out_1_3, out_1_5], dim=1)  # [batch, in_channels +32 +16, H, W]
        out_1_5_7 = torch.cat([x, out_1_3, out_1_5, out_1_7], dim=1)  # [batch, in_channels +32 +16 +8, H, W]
                
        # Second set of convolutions
        out_2_3 = F.relu(self.bn3_2(self.conv3_2(out_1_3)))  # [batch,32,H,W]
        out_2_5_2 = F.relu(self.bn5_2(self.conv5_2(out_1_3)))  # [batch,16,H,W]
        out_2_3 = torch.cat([x, out_2_3, out_2_5_2, out_1_3], dim=1)  # [batch, in_channels +32 +16 +32, H, W]
        
        out_2_5 = F.relu(self.bn5_main(self.conv5_main(out_1_3_5)))  # [batch,24,H,W]
        out_2_7 = F.relu(self.bn7_main(self.conv7_main(out_1_5_7)))  # [batch,12,H,W]
        
        out_3_5_7 = torch.cat([x, out_2_5, out_2_7], dim=1)  # [batch, in_channels +24 +12, H, W]
        out_3_b_5_7 = self.bn5_main(out_2_5)  # Use the already normalized out_2_5
        out_3_b_3 = self.bn5_2(out_2_3)        # Apply BatchNorm to out_2_3
        
        # Third set of convolutions
        out_4_3 = F.relu(self.bn3_final(self.conv3_final(out_3_5_7)))  # [batch,36,H,W]
        out_4_b_3 = self.bn3_final(out_4_3)                         # Already normalized
        
        # Final concatenation
        out = torch.cat([x, 
            out_3_b_3, 
            out_1_3, 
            out_1_3_5, 
            out_1_5_7, 
            out_3_b_5_7, 
            out_4_b_3
        ], dim=1)  # [batch, in_channels +32 +16 +24 +12 +36, H, W]
        
        out = F.relu(self.bn1x1(self.conv1x1(out)))  # [batch,24,H,W]
        out = self.pool(out)                          # [batch,24,H/2,W/2]
        
        return out

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.rescaling = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1), 
            nn.BatchNorm2d(1)
        )
        self.block1 = MK_Block(in_channels=1)
        self.block2 = MK_Block(in_channels=24)  # Output channels from block1
        self.block3 = MK_Block(in_channels=24)  # Output channels from block2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(24 * 3 * 3, 256) 
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = self.rescaling(x)
        
        x = self.block1(x)  # Output: [batch_size,24,14,14]
        x = self.block2(x)  # Output: [batch_size,24,7,7]
        x = self.block3(x)  # Output: [batch_size,24,3,3]
        
        x = self.flatten(x)  # Output: [batch_size, 24*3*3 = 216]
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)  # Output logits
        
        return x

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