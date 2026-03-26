import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Basic Residual Block
# ---------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ---------------------------
# Bottleneck Block (optional)
# ---------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d, dropout=0.0):
        super().__init__()

        width = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        return F.relu(out)


# ---------------------------
# ResNet Model
# ---------------------------
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        base_channels=64,
        norm_layer=nn.BatchNorm2d,
        dropout=0.0
    ):
        super().__init__()

        self.in_channels = base_channels

        # CIFAR-friendly stem
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(base_channels)

        # Residual stages
        self.layer1 = self._make_layer(block, base_channels, layers[0],
                                      stride=1, norm_layer=norm_layer, dropout=dropout)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1],
                                      stride=2, norm_layer=norm_layer, dropout=dropout)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2],
                                      stride=2, norm_layer=norm_layer, dropout=dropout)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3],
                                      stride=2, norm_layer=norm_layer, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride, norm_layer, dropout):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride,
                            norm_layer=norm_layer, dropout=dropout))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                norm_layer=norm_layer, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------
# Factory Functions
# ---------------------------
def resnet18_cifar(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_cifar(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_cifar(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    model = resnet18_cifar(
        num_classes=10,
        base_channels=64,
        dropout=0.1
    )

    x = torch.randn(8, 3, 32, 32)
    y = model(x)

    num_trainable_params = sum([p.numel() for p in model.parameters()])
    print('\n' + 'num_trainable_params = ' + str(num_trainable_params) + '\n')
    print("Output shape:", y.shape)  # (8, 10)