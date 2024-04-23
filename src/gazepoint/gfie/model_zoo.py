import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50


class GazeDirectionNet(nn.Module):
    def __init__(self):

        super(GazeDirectionNet, self).__init__()

        org_resnet50 = resnet50(pretrained=False)

        # backbone in the module
        self.backbone = nn.Sequential(*list(org_resnet50.children())[:-1])

        self.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, himg):
        headfeat = self.backbone(himg)
        headfeat = torch.flatten(headfeat, 1)
        gazevector = self.fc(headfeat)
        gazevector = F.normalize(gazevector)

        return gazevector


class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()

        self.gaze_direction_net = GazeDirectionNet()

    def forward(self, simg, himg):
        predicted_gazedirection = self.gaze_direction_net(himg)

        return {"pred_gazedirection": predicted_gazedirection}
