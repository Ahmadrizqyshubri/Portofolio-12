
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt

mean = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
std = torch.FloatTensor([[[0.229, 0.224, 0.225]]])

transform = transforms.Compose([
    transforms.Resize(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(filepath):
    image = Image.open(filepath).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def gram_matrix(X):
    n, c, h, w = X.shape
    X = X.view(n*c, h*w) # Flattening
    G = torch.mm(X, X.t())
    G = G.div(n*c*h*w) # Normalization
    return G

def DrawStyleImage(output):
    style_image = output[0].permute(1,2,0).cpu().detach()
    style_image = style_image*std + mean
    style_image.clamp_(0,1)

    plt.figure(figsize=(6,6))
    plt.imshow(style_image)
    plt.axis(False)
    plt.pause(0.01)
    return style_image
