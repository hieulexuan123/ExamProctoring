import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms


class DeePixBiS(nn.Module):
    def __init__(self):
        super().__init__()
        dense = models.densenet161()
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = torch.sigmoid(dec)
        out = self.linear(out_map.view(-1, 14 * 14))
        out = F.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out
    
class FAS:
    def __init__(self):
        self.model = DeePixBiS()
        self.model.load_state_dict(torch.load('model/DeePixBiS.pth', map_location='cpu'))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def detect(self, face):
        with torch.no_grad():
            face = self.transform(face)
            face = face.unsqueeze(0)
            mask, _ = self.model(face)
            res = torch.mean(mask)
        print('Score', res)

        return (res<0.5).item()