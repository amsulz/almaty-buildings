"""Grad-CAM: where on the photo the ResNet looks when it picks a class.

Takes the last block of ResNet-50 (a 7x7 grid of features), checks how much each
feature pushes the chosen class up (the gradient), and paints the grid in red
where it pushes the most. Red on the facade = good, red on trees, cars or sky =
the model learned something that is not the building.

Only val photos (Medeu), the test districts are not touched. Pictures are saved
to gradcam/ (not on GitHub, because they contain the Street View photos):
  python src/gradcam.py
"""
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from train import CLASSES, MODEL_PATH, PHOTOS_DIR, ROOT, device, eval_tf, load_split

OUT_DIR = os.path.join(ROOT, "gradcam")


def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()


def gradcam(model, img):
    """Returns the predicted class, its probability and a heat map 0..1 the size of the photo."""
    saved = {}
    hook = model.layer4.register_forward_hook(lambda m, i, out: saved.update(features=out))
    x = eval_tf(img).unsqueeze(0).to(device)
    out = model(x)
    hook.remove()

    k = out.argmax(1).item()
    features = saved["features"]
    grad = torch.autograd.grad(out[0, k], features)[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)  # one number per feature map
    cam = torch.relu((weights * features).sum(1))[0]  # only what pushes the class up
    cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()

    heat = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)
    prob = torch.softmax(out, 1)[0, k].item()
    return CLASSES[k], prob, np.asarray(heat) / 255


def overlay(img, heat):
    """Photo with the heat map on top: red = important, no colour = not important."""
    photo = np.asarray(img, dtype=float)
    red = np.zeros_like(photo)
    red[..., 0] = 255
    alpha = 0.5 * heat[..., None]
    return Image.fromarray(np.uint8(photo * (1 - alpha) + red * alpha))


def main():
    _, val, _ = load_split()
    model = load_model()
    os.makedirs(OUT_DIR, exist_ok=True)
    right = 0
    for row in val.itertuples():
        img = Image.open(os.path.join(PHOTOS_DIR, row.photo)).convert("RGB")
        pred, prob, heat = gradcam(model, img)
        right += pred == row.label
        # file name says the true label and the answer, so mistakes are easy to find
        name = f"{row.label}_as_{pred}_{prob:.2f}_{row.photo}"
        side_by_side = Image.new("RGB", (img.width * 2, img.height))
        side_by_side.paste(img, (0, 0))
        side_by_side.paste(overlay(img, heat), (img.width, 0))
        side_by_side.save(os.path.join(OUT_DIR, name))
    print(f"{len(val)} val photos, {right} right, pictures in {OUT_DIR}")


if __name__ == "__main__":
    main()
