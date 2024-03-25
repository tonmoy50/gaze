import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, expansion, is_Bottleneck, stride
    ):
        super(Bottleneck, self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck

        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                )
            )
            projection_layer.append(
                nn.BatchNorm2d(self.intermediate_channels * self.expansion)
            )
            self.projection = nn.Sequential(*projection_layer)

        self.relu = nn.ReLU()
        if self.is_Bottleneck:
            # 1x1
            self.conv1_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)

            # 3x3
            self.conv2_3x3 = nn.Conv2d(
                in_channels=self.intermediate_channels,
                out_channels=self.intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)

            # 1x1
            self.conv3_1x1 = nn.Conv2d(
                in_channels=self.intermediate_channels,
                out_channels=self.intermediate_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.batchnorm3 = nn.BatchNorm2d(
                self.intermediate_channels * self.expansion
            )

        else:
            # 3x3
            self.conv1_3x3 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)

            # 3x3
            self.conv2_3x3 = nn.Conv2d(
                in_channels=self.intermediate_channels,
                out_channels=self.intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)

    def forward(self, x):
        # input stored to be added before the final relu
        in_x = x

        if self.is_Bottleneck:
            x = self.relu(self.batchnorm1(self.conv1_1x1(x)))
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))
            x = self.batchnorm3(self.conv3_1x1(x))

        else:
            x = self.relu(self.batchnorm1(self.conv1_3x3(x)))
            x = self.batchnorm2(self.conv2_3x3(x))

        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet, self).__init__()

        self.channels_list = [64, 128, 256, 512]
        self.repeatition_list = [3, 4, 6, 3]
        self.expansion = 4
        self.is_Bottleneck = True

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._block(
            64,
            self.channels_list[0],
            self.repeatition_list[0],
            self.expansion,
            self.is_Bottleneck,
            stride=1,
        )
        self.block2 = self._block(
            self.channels_list[0] * self.expansion,
            self.channels_list[1],
            self.repeatition_list[1],
            self.expansion,
            self.is_Bottleneck,
            stride=2,
        )
        self.block3 = self._block(
            self.channels_list[1] * self.expansion,
            self.channels_list[2],
            self.repeatition_list[2],
            self.expansion,
            self.is_Bottleneck,
            stride=2,
        )
        self.block4 = self._block(
            self.channels_list[2] * self.expansion,
            self.channels_list[3],
            self.repeatition_list[3],
            self.expansion,
            self.is_Bottleneck,
            stride=2,
        )

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

    def _block(
        self,
        in_channels,
        intermediate_channels,
        num_repeat,
        expansion,
        is_Bottleneck,
        stride,
    ):
        layers = []

        layers.append(
            Bottleneck(
                in_channels,
                intermediate_channels,
                expansion,
                is_Bottleneck,
                stride=stride,
            )
        )
        for num in range(1, num_repeat):
            layers.append(
                Bottleneck(
                    intermediate_channels * expansion,
                    intermediate_channels,
                    expansion,
                    is_Bottleneck,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # bottom up
        # self.resnet = resnet_fpn.resnet50(pretrained=True)
        self.resnet = ResNet()

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.c5_conv = nn.Conv2d(2048, 256, (1, 1))
        self.c4_conv = nn.Conv2d(1024, 256, (1, 1))
        self.c3_conv = nn.Conv2d(512, 256, (1, 1))
        self.c2_conv = nn.Conv2d(256, 256, (1, 1))
        # self.max_pool = nn.MaxPool2d((1, 1), stride=2)

        self.p5_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p4_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p3_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p2_conv = nn.Conv2d(256, 256, (3, 3), padding=1)

        # predict heatmap
        self.sigmoid = nn.Sigmoid()
        self.predict = nn.Conv2d(256, 1, (3, 3), padding=1)

    def top_down(self, x):
        c2, c3, c4, c5 = x
        p5 = self.c5_conv(c5)
        p4 = self.upsample(p5) + self.c4_conv(c4)
        p3 = self.upsample(p4) + self.c3_conv(c3)
        p2 = self.upsample(p3) + self.c2_conv(c2)

        p5 = self.relu(self.p5_conv(p5))
        p4 = self.relu(self.p4_conv(p4))
        p3 = self.relu(self.p3_conv(p3))
        p2 = self.relu(self.p2_conv(p2))

        return p2, p3, p4, p5

    def forward(self, x):
        # bottom up
        c2, c3, c4, c5 = self.resnet(x)

        # top down
        p2, p3, p4, p5 = self.top_down((c2, c3, c4, c5))

        heatmap = self.sigmoid(self.predict(p2))
        return heatmap


class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        # self.face_net = M.resnet50(pretrained=True)
        self.face_net = ResNet()
        self.face_process = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True))

        self.fpn_net = FPN()

        self.eye_position_transform = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256), nn.ReLU(inplace=True), nn.Linear(256, 2)
        )

        self.relu = nn.ReLU(inplace=False)

        # change first conv layer for fpn_net because we concatenate
        # multi-scale gaze field with image image
        conv = [x.clone() for x in self.fpn_net.resnet.conv1.parameters()][0]
        new_kernel_channel = conv.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        new_kernel = torch.cat((conv.data, new_kernel_channel), 1)
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = new_kernel
        self.fpn_net.resnet.conv1 = new_conv

    def forward(self, x):
        image, face_image, gaze_field, eye_position = x
        # face part forward
        face_feature = self.face_net(face_image)
        face_feature = self.face_process(face_feature)

        # eye position transform
        eye_feature = self.eye_position_transform(eye_position)

        # fusion
        feature = torch.cat((face_feature, eye_feature), 1)
        direction = self.fusion(feature)

        # infer gaze direction and normalized
        norm = torch.norm(direction, 2, dim=1)
        normalized_direction = direction / norm.view([-1, 1])

        # generate gaze field map
        batch_size, channel, height, width = gaze_field.size()
        gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        gaze_field = gaze_field.view([batch_size, -1, 2])
        gaze_field = torch.matmul(
            gaze_field, normalized_direction.view([batch_size, 2, 1])
        )
        gaze_field_map = gaze_field.view([batch_size, height, width, 1])
        gaze_field_map = gaze_field_map.permute([0, 3, 1, 2]).contiguous()

        gaze_field_map = self.relu(gaze_field_map)
        # print gaze_field_map.size()

        # mask with gaze_field
        gaze_field_map_2 = torch.pow(gaze_field_map, 2)
        gaze_field_map_3 = torch.pow(gaze_field_map, 5)
        image = torch.cat(
            [image, gaze_field_map, gaze_field_map_2, gaze_field_map_3], dim=1
        )
        heatmap = self.fpn_net(image)

        return direction, heatmap
