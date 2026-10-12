import torch
import torch.nn as nn
from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.BatchNorm2d(in_features // 8),
            nn.ReLU(),
            nn.Conv2d(in_features // 8, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super(CropDiseaseModel, self).__init__()
        # 1. Load pre-trained VGG16
        vgg16 = models.vgg16(weights='DEFAULT')
        self.vgg16_features = vgg16.features
        
        # 2. Add Spatial Attention Mechanism after VGG16 features
        self.attention = AttentionBlock(512)
        self.vgg16_avgpool = vgg16.avgpool
        
        # 3. Load pre-trained MobileNetV2
        mobilenet_v2 = models.mobilenet_v2(weights='DEFAULT')
        self.mobilenet_features = mobilenet_v2.features
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace classifier (Combine VGG16 features and MobileNetV2 features)
        # VGG16 outputs: 512 channels * 7 * 7 = 25088 flattened features
        # MobileNetV2 outputs: 1280 channels * 1 * 1 = 1280 flattened features
        # Total concatenated features = 25088 + 1280 = 26368
        self.classifier = nn.Sequential(
            nn.Linear(26368, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # --- VGG16 + Attention Branch ---
        vgg_x = self.vgg16_features(x)
        vgg_x = self.attention(vgg_x)
        vgg_x = self.vgg16_avgpool(vgg_x)
        vgg_x = torch.flatten(vgg_x, 1)
        
        # --- MobileNetV2 Branch ---
        mn_x = self.mobilenet_features(x)
        mn_x = self.mobilenet_avgpool(mn_x)
        mn_x = torch.flatten(mn_x, 1)
        
        # --- Combine Branches ---
        combined_x = torch.cat((vgg_x, mn_x), dim=1)
        
        # --- Classifier ---
        out = self.classifier(combined_x)
        return out
