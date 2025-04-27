# model.py (PyTorch version)

import torch
import torch.nn as nn
import torchvision.models as models

class InceptionResNetV2Loader(nn.Module):
    def __init__(self):
        super(InceptionResNetV2Loader, self).__init__()
        # Pretrained InceptionResNetV2은 torchvision에 없음. InceptionV3 사용 또는 외부 소스 사용
        inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.backbone = nn.Sequential(*list(inception.children())[:-1])  # 마지막 FC 제거

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class DiagnoseModel(nn.Module):
    def __init__(self):
        super(DiagnoseModel, self).__init__()
        self.fc = nn.Linear(3, 128)  # gender, age_group, image_number를 입력으로 받음
        self.relu = nn.ReLU()

    def forward(self, inputs):
        gender, age_group, image_number = inputs
        x = torch.cat([gender.unsqueeze(1), age_group.unsqueeze(1), image_number.unsqueeze(1)], dim=1)
        x = self.fc(x)
        x = self.relu(x)
        return x

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        self.cnn = InceptionResNetV2Loader()
        self.tabular = DiagnoseModel()
        self.fc1 = nn.Linear(2048 + 128, 256)  # InceptionV3 output + tabular feature
        self.fc2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        img, tab = inputs
        x1 = self.cnn(img)
        x2 = self.tabular(tab)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
