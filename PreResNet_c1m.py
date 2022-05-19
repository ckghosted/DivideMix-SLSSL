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

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
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

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        downsample = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += downsample
        return out

class PreActBlock32(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock32, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        downsample = self.downsample(x)
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        out = x + downsample
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_backup(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
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

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        downsample = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += downsample
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, weights=None, get_feat=None, is_training=False):
        if weights==None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            feat = x.view(x.size(0), -1)
            x = self.fc(feat)
            if get_feat:
                return x,feat
            else:
                return x
        else:
                        
            x = F.conv2d(x, weights['conv1.weight'], stride=2, padding=3)
            x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, weights['bn1.weight'], weights['bn1.bias'],training=is_training)            
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            #layer 1
            for i in range(3):
                residual = x
                out = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i],training=is_training)      
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i],training=is_training)     
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer1.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer1[i].bn3.running_mean, self.layer1[i].bn3.running_var, 
                                 weights['layer1.%d.bn3.weight'%i], weights['layer1.%d.bn3.bias'%i],training=is_training)                               
                if i==0:
                    residual = F.conv2d(x, weights['layer1.%d.downsample.0.weight'%i], stride=1)  
                    residual = F.batch_norm(residual, self.layer1[i].downsample[1].running_mean, self.layer1[i].downsample[1].running_var, 
                                 weights['layer1.%d.downsample.1.weight'%i], weights['layer1.%d.downsample.1.bias'%i],training=is_training)  
                x = out + residual     
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 2
            for i in range(4):
                residual = x
                out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i],training=is_training)     
                out = F.threshold(out, 0, 0, inplace=True)
                if i==0:
                    out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], stride=2, padding=1)
                else:
                    out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i],training=is_training)    
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer2.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer2[i].bn3.running_mean, self.layer2[i].bn3.running_var, 
                                 weights['layer2.%d.bn3.weight'%i], weights['layer2.%d.bn3.bias'%i],training=is_training)                    
                if i==0:
                    residual = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], stride=2)  
                    residual = F.batch_norm(residual, self.layer2[i].downsample[1].running_mean, self.layer2[i].downsample[1].running_var, 
                                 weights['layer2.%d.downsample.1.weight'%i], weights['layer2.%d.downsample.1.bias'%i],training=is_training)  
                x = out + residual  
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 3
            for i in range(6):
                residual = x
                out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i],training=is_training)   
                out = F.threshold(out, 0, 0, inplace=True)
                if i==0:
                    out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], stride=2, padding=1)
                else:
                    out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i],training=is_training)     
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer3.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer3[i].bn3.running_mean, self.layer3[i].bn3.running_var, 
                                 weights['layer3.%d.bn3.weight'%i], weights['layer3.%d.bn3.bias'%i],training=is_training)                    
                if i==0:
                    residual = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], stride=2)  
                    residual = F.batch_norm(residual, self.layer3[i].downsample[1].running_mean, self.layer3[i].downsample[1].running_var, 
                                 weights['layer3.%d.downsample.1.weight'%i], weights['layer3.%d.downsample.1.bias'%i],training=is_training)  
                x = out + residual    
                x = F.threshold(x, 0, 0, inplace=True)
                
            #layer 4
            for i in range(3):
                residual = x
                out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                                 weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i],training=is_training)   
                out = F.threshold(out, 0, 0, inplace=True)
                if i==0:
                    out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], stride=2, padding=1)
                else:
                    out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                                 weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i],training=is_training)   
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer4.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer4[i].bn3.running_mean, self.layer4[i].bn3.running_var, 
                                 weights['layer4.%d.bn3.weight'%i], weights['layer4.%d.bn3.bias'%i],training=is_training)                    
                if i==0:
                    residual = F.conv2d(x, weights['layer4.%d.downsample.0.weight'%i], stride=2)  
                    residual = F.batch_norm(residual, self.layer4[i].downsample[1].running_mean, self.layer4[i].downsample[1].running_var, 
                                 weights['layer4.%d.downsample.1.weight'%i], weights['layer4.%d.downsample.1.bias'%i],training=is_training)  
                x = out + residual    
                x = F.threshold(x, 0, 0, inplace=True)
                
            x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
            x = x.view(x.size(0), -1)
            x = F.linear(x, weights['fc.weight'], weights['fc.bias'])                
            return x
    
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

class ResNet_backup(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, weights=None, is_training=False, lin=0, lout=5, get_feat=False):
        if weights==None:
            if get_feat:
                # ignore lin and lout
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                feat = torch.flatten(x, 1)
                out = self.fc(feat)
                return out, feat
            else:
                out = x
                if lin < 1 and lout > -1:
                    out = self.conv1(out)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.maxpool(out)
                if lin < 2 and lout > 0:
                    out = self.layer1(out)
                if lin < 3 and lout > 1:
                    out = self.layer2(out)
                if lin < 4 and lout > 2:
                    out = self.layer3(out)
                if lin < 5 and lout > 3:
                    out = self.layer4(out)
                if lout > 4:
                    out = self.avgpool(out)
                    out = torch.flatten(out, 1)
                    out = self.fc(out)
                return out
        else:
            x = F.conv2d(x, weights['conv1.weight'], stride=2, padding=3)
            x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, 
                             weights['bn1.weight'], weights['bn1.bias'], training=is_training)
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            #layer 1
            strides = [1, 1, 1]
            for i in range(3):
                if 'layer1.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer1.%d.downsample.0.weight'%i], stride=strides[i])
                    x = F.batch_norm(x, self.layer1[i].downsample[1].running_mean, self.layer1[i].downsample[1].running_var, 
                                     weights['layer1.%d.downsample.1.weight'%i], weights['layer1.%d.downsample.1.bias'%i], training=is_training)
                else:
                    downsample = x
                x = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=strides[i], padding=1)
                x = F.batch_norm(x, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)

                x = F.conv2d(x, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                
                x = F.conv2d(x, weights['layer1.%d.conv3.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer1[i].bn3.running_mean, self.layer1[i].bn3.running_var, 
                                 weights['layer1.%d.bn3.weight'%i], weights['layer1.%d.bn3.bias'%i], training=is_training)
                
                x = x + downsample
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 2
            strides = [2, 1, 1, 1]
            for i in range(4):
                if 'layer2.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], stride=strides[i])
                    x = F.batch_norm(x, self.layer2[i].downsample[1].running_mean, self.layer2[i].downsample[1].running_var, 
                                     weights['layer2.%d.downsample.1.weight'%i], weights['layer2.%d.downsample.1.bias'%i], training=is_training)
                else:
                    downsample = x
                x = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=strides[i], padding=1)
                x = F.batch_norm(x, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)

                x = F.conv2d(x, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                
                x = F.conv2d(x, weights['layer2.%d.conv3.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer2[i].bn3.running_mean, self.layer2[i].bn3.running_var, 
                                 weights['layer2.%d.bn3.weight'%i], weights['layer2.%d.bn3.bias'%i], training=is_training)
                
                x = x + downsample
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 3
            strides = [2, 1, 1, 1, 1, 1]
            for i in range(6):
                if 'layer3.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], stride=strides[i])
                    x = F.batch_norm(x, self.layer3[i].downsample[1].running_mean, self.layer3[i].downsample[1].running_var, 
                                     weights['layer3.%d.downsample.1.weight'%i], weights['layer3.%d.downsample.1.bias'%i], training=is_training)
                else:
                    downsample = x
                x = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=strides[i], padding=1)
                x = F.batch_norm(x, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)

                x = F.conv2d(x, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                
                x = F.conv2d(x, weights['layer3.%d.conv3.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer3[i].bn3.running_mean, self.layer3[i].bn3.running_var, 
                                 weights['layer3.%d.bn3.weight'%i], weights['layer3.%d.bn3.bias'%i], training=is_training)
                
                x = x + downsample
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 4
            strides = [2, 1, 1]
            for i in range(3):
                if 'layer4.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer4.%d.downsample.0.weight'%i], stride=strides[i])
                    x = F.batch_norm(x, self.layer4[i].downsample[1].running_mean, self.layer4[i].downsample[1].running_var, 
                                     weights['layer4.%d.downsample.1.weight'%i], weights['layer4.%d.downsample.1.bias'%i], training=is_training)
                else:
                    downsample = x
                x = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], stride=strides[i], padding=1)
                x = F.batch_norm(x, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                                 weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)

                x = F.conv2d(x, weights['layer4.%d.conv2.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                                 weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                
                x = F.conv2d(x, weights['layer4.%d.conv3.weight'%i], stride=1, padding=1)
                x = F.batch_norm(x, self.layer4[i].bn3.running_mean, self.layer4[i].bn3.running_var, 
                                 weights['layer4.%d.bn3.weight'%i], weights['layer4.%d.bn3.bias'%i], training=is_training)
                
                x = x + downsample
                x = F.threshold(x, 0, 0, inplace=True)
            x = F.avg_pool2d(x, kernel_size=4)
            feat = x.view(x.size(0), -1)
            out = F.linear(feat, weights['fc.weight'], weights['fc.bias'])                
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
        self.layer1 = self._make_layer(block=PreActBlock32, planes=32, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(block=PreActBlock32, planes=64, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(block=PreActBlock32, planes=128, num_blocks=5, stride=2)
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
                if 'layer1.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer1.%d.downsample.0.weight'%i], stride=strides[i])
                else:
                    downsample = x
                x = F.batch_norm(x, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + downsample
            #layer 2
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer2.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], stride=strides[i])
                else:
                    downsample = x
                x = F.batch_norm(x, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + downsample
            #layer 3
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer3.%d.downsample.0.weight'%i in weights.keys():
                    downsample = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], stride=strides[i])
                else:
                    downsample = x
                x = F.batch_norm(x, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + downsample
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
