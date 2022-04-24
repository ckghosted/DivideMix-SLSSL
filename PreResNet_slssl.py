import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, weights=None, is_training=False, lin=0, lout=5):
        if weights==None:
            out = x
            if lin < 1 and lout > -1:
                out = self.conv1(out)
                out = self.bn1(out)
                out = F.relu(out)
            if lin < 2 and lout > 0:
                out = self.layer1(out)
            if lin < 3 and lout > 1:
                out = self.layer2(out)
            if lin < 4 and lout > 2:
                out = self.layer3(out)
            if lin < 5 and lout > 3:
                out = self.layer4(out)
            if lout > 4:
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
            return out
        else:
            x = F.conv2d(x, weights['conv1.weight'], stride=1, padding=1)
            x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, 
                             weights['bn1.weight'], weights['bn1.bias'], training=is_training)
            x = F.threshold(x, 0, 0, inplace=True)
            #layer 1
            strides = [1, 1]
            for i in range(2):
                if 'layer1.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer1.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 2
            strides = [2, 1]
            for i in range(2):
                if 'layer2.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer2.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 3
            strides = [2, 1]
            for i in range(2):
                if 'layer3.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer3.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 4
            strides = [2, 1]
            for i in range(2):
                if 'layer4.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer4.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                                 weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                                 weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer4.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            x = F.avg_pool2d(x, kernel_size=4)
            feat = x.view(x.size(0), -1)
            out = F.linear(feat, weights['linear.weight'], weights['linear.bias'])                
            return out

    def save_BN_state_dict(self):
        bn_state = {}
        for name, param in self.state_dict().items():
            if 'bn' in name and 'running' in name:
                # print('extracting %s...' % name)
                bn_state[name] = torch.randn_like(param)
                bn_state[name].copy_(param)
        return bn_state
    
    def load_BN_state_dict(self, state_dict): 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'bn' in name and 'running' in name:
                # print('loading %s...' % name)
                own_state[name].copy_(param)

def ResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

class PreActResNet(nn.Module):
    def __init__(self):
        super(PreActResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block=PreActBlock, planes=32, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(block=PreActBlock, planes=64, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(block=PreActBlock, planes=128, num_blocks=5, stride=2)
        self.bn_final = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(128, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # strides = [stride, 1, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, weights=None, get_feat=None, is_training=True):
        if weights==None:
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = F.relu(self.bn_final(x))
            x = self.avgpool(x)
            feat = x.view(x.size(0), -1)
            out = self.linear(feat)
            if get_feat:
                return out,feat
            else:
                return out
        else:
            x = F.conv2d(x, weights['conv1.weight'], stride=1, padding=1)
            #layer 1
            strides = [1, 1, 1, 1, 1]
            for i in range(5):
                if 'layer1.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer1.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 2
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer2.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer2.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 3
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer3.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer3.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            x = F.batch_norm(x, self.bn_final.running_mean, self.bn_final.running_var,
                             weights['bn_final.weight'], weights['bn_final.bias'], training=is_training)            
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.avg_pool2d(x, kernel_size=8, stride=1, padding=0)
            feat = x.view(x.size(0), -1)
            out = F.linear(feat, weights['linear.weight'], weights['linear.bias'])                
            return out
    
    def save_BN_state_dict(self):
        bn_state = {}
        for name, param in self.state_dict().items():
            if 'bn' in name and 'running' in name:
                # print('extracting %s...' % name)
                bn_state[name] = torch.randn_like(param)
                bn_state[name].copy_(param)
        return bn_state
    
    def load_BN_state_dict(self, state_dict): 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'bn' in name and 'running' in name:
                # print('loading %s...' % name)
                own_state[name].copy_(param)

def PreActResNet32():
    return PreActResNet()
