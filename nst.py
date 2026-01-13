import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache VGG features globally (avoid reloading every run)
_VGG = None
def get_vgg_features():
    global _VGG
    if _VGG is None:
        _VGG = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    return _VGG

def load_image(path, size=256):
    image = Image.open(path).convert("RGB")
    transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return transform(image).unsqueeze(0).to(device)

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    return torch.mm(features, features.t()).div(b * c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0, device=target.device)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0, device=target_feature.device)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def build_model_and_losses(cnn, style_img, content_img, style_layers, content_layers):
    normalization = Normalization().to(device)
    content_losses, style_losses = [], []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            name = layer.__class__.__name__.lower()
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f'content_loss_{i}', cl)
            content_losses.append(cl)

        if name in style_layers:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f'style_loss_{i}', sl)
            style_losses.append(sl)

    # Trim after last loss layer
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    return model[:j+1], style_losses, content_losses

# --- Faster Classic NST with Adam ---
def run_style_transfer_adam(content_path, style_path, size=256, steps=50,
                            style_weight=1e6, content_weight=1.0, lr=0.02):
    content_img = load_image(content_path, size=size)
    style_img = load_image(style_path, size=size)
    input_img = content_img.clone().requires_grad_(True)

    cnn = get_vgg_features()
    style_layers = {'conv_1','conv_2','conv_3','conv_4','conv_5'}
    content_layers = {'conv_4'}

    model, style_losses, content_losses = build_model_and_losses(
        cnn, style_img, content_img, style_layers, content_layers
    )

    optimizer = optim.Adam([input_img], lr=lr)
    style_hist, content_hist, total_hist = [], [], []

    for step in range(steps):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            input_img.clamp_(0, 1)

        style_hist.append(float(style_score.item()))
        content_hist.append(float(content_score.item()))
        total_hist.append(float(loss.item()))

    return input_img.detach().cpu(), style_hist, content_hist, total_hist

# --- Fast Style Transfer (fewer steps) ---
def run_fast_style_transfer(content_path, style_path, size=256,
                            style_weight=1e6, content_weight=1.0):
    return run_style_transfer_adam(content_path, style_path, size=size,
                                   steps=20, style_weight=style_weight,
                                   content_weight=content_weight, lr=0.03)

# --- AdaIN-like pixel approximation ---
def run_adain_pixel(content_path, style_path, size=256):
    c_img = Image.open(content_path).convert("RGB").resize((size, size))
    s_img = Image.open(style_path).convert("RGB").resize((size, size))
    c = np.asarray(c_img).astype(np.float32) / 255.0
    s = np.asarray(s_img).astype(np.float32) / 255.0

    c_mean, c_std = c.mean(axis=(0,1), keepdims=True), c.std(axis=(0,1), keepdims=True)+1e-6
    s_mean, s_std = s.mean(axis=(0,1), keepdims=True), s.std(axis=(0,1), keepdims=True)+1e-6

    normalized = (c - c_mean) / c_std
    adain = normalized * s_std + s_mean
    adain = np.clip(adain, 0.0, 1.0)

    return adain, [], [], []