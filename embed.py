import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def load_embed():
    return


def load_embed_docs(dir):
    return
    

def load_embed_pics(dir):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    dir = os.path.join('data', dir, 'pic1.jpg')
    img = Image.open(dir)
    img = torch.unsqueeze(transform(img), dim=0)
    with torch.no_grad():
        feats = model(img)
    print(feats)


if __name__ == "__main__":
    dir = 'test'
    load_embed_pics(dir)