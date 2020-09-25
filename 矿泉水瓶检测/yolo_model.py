import ResNet
import torch.nn as nn
import torch


yolo_output = 24  # 5*2+14

class YOLOV0(nn.Module):
    def __init__(self):
        super(YOLOV0, self).__init__()
        resnet= ResNet.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu =resnet.relu
        self.maxpool=resnet.maxpool
        self.layer1=resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))
        # 决策层：检测层
        self.detector = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096,1470),
            nn.Linear(4096, 24*14*14),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.detector(x)
        b, _ = x.shape
        x = x.view(b, 14, 14, 24)
        return x


if __name__ == '__main__':
    resnet = ResNet.resnet18()
    x = torch.randn(1, 3, 512, 512)
    yolov0 = YOLOV0()
