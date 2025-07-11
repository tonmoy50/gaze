import torch
import torch.nn as nn
import torch.nn.functional as F
from gfiemodel.resnet import resnet50


class Encoder(nn.Module):
    """Encoder in the Module for Generating GazeHeatmap"""

    def __init__(self, pretrained=False):

        super(Encoder, self).__init__()

        org_resnet = resnet50(pretrained)

        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = org_resnet.bn1
        self.relu = org_resnet.relu
        self.maxpool = org_resnet.maxpool
        self.layer1 = org_resnet.layer1
        self.layer2 = org_resnet.layer2
        self.layer3 = org_resnet.layer3
        self.layer4 = org_resnet.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Decoder(nn.Module):
    """Decoder in the Module for Generating GazeHeatmap"""

    def __init__(self):
        super(Decoder, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.compress_conv1 = nn.Conv2d(
            2048, 1024, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(
            1024, 512, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.compress_bn2 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.relu(x)
        x = self.compress_conv2(x)
        x = self.compress_bn2(x)
        x = self.relu(x)

        x = self.deconv1(x)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x


# class Decoder(nn.Module):
#     """Decoder in the Module for Generating GazeHeatmap with 128x128 output"""

#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.relu = nn.ReLU(inplace=True)

#         self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
#         self.compress_bn1 = nn.BatchNorm2d(1024)
#         self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1)
#         self.compress_bn2 = nn.BatchNorm2d(512)

#         # Carefully designed deconvs to reach 128x128 from 7x7
#         self.deconv1 = nn.ConvTranspose2d(
#             512, 256, kernel_size=4, stride=2, padding=1
#         )  # 7 → 14
#         self.deconv_bn1 = nn.BatchNorm2d(256)
#         self.deconv2 = nn.ConvTranspose2d(
#             256, 128, kernel_size=4, stride=2, padding=1
#         )  # 14 → 28
#         self.deconv_bn2 = nn.BatchNorm2d(128)
#         self.deconv3 = nn.ConvTranspose2d(
#             128, 64, kernel_size=4, stride=2, padding=1
#         )  # 28 → 56
#         self.deconv_bn3 = nn.BatchNorm2d(64)
#         self.deconv4 = nn.ConvTranspose2d(
#             64, 32, kernel_size=4, stride=2, padding=1
#         )  # 56 → 112
#         self.deconv_bn4 = nn.BatchNorm2d(32)
#         self.deconv5 = nn.ConvTranspose2d(
#             32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
#         )  # 112 → 224

#         # Reduce back to 128x128 with interpolation
#         self.resize = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False)
#         self.final_conv = nn.Conv2d(1, 1, kernel_size=1)

#     def forward(self, x):
#         x = self.relu(self.compress_bn1(self.compress_conv1(x)))
#         x = self.relu(self.compress_bn2(self.compress_conv2(x)))

#         x = self.relu(self.deconv_bn1(self.deconv1(x)))  # 14
#         x = self.relu(self.deconv_bn2(self.deconv2(x)))  # 28
#         x = self.relu(self.deconv_bn3(self.deconv3(x)))  # 56
#         x = self.relu(self.deconv_bn4(self.deconv4(x)))  # 112
#         x = self.relu(self.deconv5(x))  # 224

#         x = self.resize(x)  # resize to 128x128
#         x = self.final_conv(x)
#         return x


class EGDModule(nn.Module):
    """The Module for Estimating Gaze Direction"""

    def __init__(self):

        super(EGDModule, self).__init__()

        org_resnet50 = resnet50(pretrained=False)

        # backbone in the module
        self.backbone = nn.Sequential(*list(org_resnet50.children())[:-1])

        self.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, himg):
        """
        Args:
            himg: cropped head image
        Returns:
            gazevector: normalized gaze direction predicted by the module.
        """
        headfeat = self.backbone(himg)
        headfeat = torch.flatten(headfeat, 1)
        gazevector = self.fc(headfeat)
        gazevector = F.normalize(gazevector, p=2, dim=1)

        return gazevector


class PSFoVModule(nn.Module):
    """The Module for Perceiving Stero FoV"""

    def __init__(self):

        super(PSFoVModule, self).__init__()

        self.relu = nn.ReLU()
        self.alpha = 3

    def forward(self, matrix_T, gazevector, simg_shape):
        """
        Args:
            matrix_T: unprojected coordiantes represent by matrix T
            gazevector: normalized gaze direction predicted by the EGD module.
            simg_shape: the shape of the simg
        Returns:
            SFoVheatmap: F and F' in paper
        Notes:
             for convenience, depthmap->matrix_T (Eq. 3) in paper is implemented in the data processing.
        """
        bs = matrix_T.shape[0]
        h, w = simg_shape[2:]

        gazevector = gazevector.unsqueeze(2)

        matrix_T = matrix_T.reshape([bs, -1, 3])

        F = torch.matmul(matrix_T, gazevector)
        F = F.reshape([bs, 1, h, w])

        F = self.relu(F)
        F_alpha = torch.pow(F, self.alpha)

        SFoVheatmap = [F, F_alpha]

        return SFoVheatmap


class GGHModule(nn.Module):
    """The Module for Generating GazeHeatmap"""

    def __init__(self, pretrained=False):

        super(GGHModule, self).__init__()

        self.encoder = Encoder(pretrained=pretrained)
        self.decoder = Decoder()

    def forward(self, simg, SFoVheatmap, headloc):
        """
        Args:
              simg: scene image
              SFoVheatmap: Stereo FoV heatmap generated by the Module for Perceiving Stero FoV
              headloc: mask representing the position of the head in scene image
        Returns:
              gazeheatmap: A heatmap representing a 2D gaze target
        """

        input = torch.cat([simg, headloc] + SFoVheatmap, dim=1)

        global_feat = self.encoder(input)

        gazeheatmap = self.decoder(global_feat)

        return gazeheatmap
