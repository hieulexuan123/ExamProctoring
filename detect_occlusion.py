import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import cv2
from PIL import Image
from torchvision.transforms import transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = mobilenet_v2().features
        self.head = nn.Sequential(
            nn.Linear(1280, 128, bias=True),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1, bias=True)
        )
    
    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = F.avg_pool2d(x, kernel_size=7)
        x = torch.flatten(x, start_dim=1)
        out = self.head(x)
        out = out.squeeze(-1)
        
        return out

class OcclusionDetector:
    def __init__(self):
        self.model = Net()
        self.model.load_state_dict(torch.load('model/occlussion_classifier.pt', map_location='cpu'))
        self.model.eval()
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

    def detect(self, face):
        face = Image.fromarray(face) #convert to PIL image
        face = self.transform(face)
        face = face.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(face)
            print(pred)
            pred = torch.sigmoid(pred)
        return (pred>0.5).item()

