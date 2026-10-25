"""Fine-tune a ResNet-50 (pretrained on ImageNet) to tell panel / brick / monolith.

Split by district, so the model is tested on parts of the city it has never seen:
  test  = Almaly + Jetisu (the same 106 photos the other AIs got)
  val   = Medeu (to choose the best epoch)
  train = all other districts
Saves the model to models/resnet50.pt and test answers to results/answers_resnet.csv
(same format as the Codex answers, so src/compare_llm.py can score it).
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = os.path.join(os.path.dirname(__file__), "..")
PHOTOS_DIR = os.path.join(ROOT, "data", "photos")
MODEL_PATH = os.path.join(ROOT, "models", "resnet50.pt")
ANSWERS_PATH = os.path.join(ROOT, "results", "answers_resnet.csv")

CLASSES = ["panel", "brick", "monolith"]
TEST_DISTRICTS = ["Almaly District", "Jetisu district"]
VAL_DISTRICTS = ["Медеу ауданы"]
EPOCHS = 15
BATCH = 16
LR = 1e-4

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet colours, because the network was pretrained on ImageNet
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# small random changes, so the model doesn't memorise exact photos
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), norm])


class Photos(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(PHOTOS_DIR, row.photo)).convert("RGB")
        return self.tf(img), CLASSES.index(row.label)


def load_split():
    labels = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
    labels["osm_id"] = labels.photo.str[:-4].astype(int)
    labels["label"] = labels.label.replace({"tile": "brick"})  # 5 = brick under facing
    labels = labels[labels.label.isin(CLASSES)]
    df = labels.merge(pd.read_csv(os.path.join(ROOT, "data", "districts.csv")), on="osm_id")
    test = df[df.district.isin(TEST_DISTRICTS)]
    val = df[df.district.isin(VAL_DISTRICTS)]
    train = df[~df.district.isin(TEST_DISTRICTS + VAL_DISTRICTS)]
    return train, val, test


def predict(model, df):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(Photos(df, eval_tf), batch_size=BATCH):
            preds += model(x.to(device)).argmax(1).tolist()
    return [CLASSES[p] for p in preds]


def main():
    train, val, test = load_split()
    print(f"train {len(train)}, val {len(val)}, test {len(test)}, device {device}")
    print(train.label.value_counts().to_dict())

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # new last layer: 3 classes
    model = model.to(device)

    # brick is rare, so its mistakes cost more (weight = 1 / how often the class appears)
    counts = train.label.value_counts()
    weights = torch.tensor([len(train) / counts[c] for c in CLASSES], dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(Photos(train, train_tf), batch_size=BATCH, shuffle=True)

    best = -1
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        val_acc = (pd.Series(predict(model, val)) == val.label.values).mean()
        print(f"epoch {epoch:2d}  loss {total / len(train):.3f}  val acc {val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    # test only once, with the best epoch on val
    model.load_state_dict(torch.load(MODEL_PATH))
    test = test.copy()
    test["pred"] = predict(model, test)
    print(f"best val acc {best:.3f}, test acc {(test.pred == test.label).mean():.3f}")

    key = pd.read_csv(os.path.join(ROOT, "data", "llm_test_key.csv"), dtype={"number": str})
    answers = key[["number", "photo"]].merge(test[["photo", "pred"]], on="photo")
    answers.rename(columns={"number": "photo", "photo": "file", "pred": "label"})[
        ["photo", "label"]].to_csv(ANSWERS_PATH, index=False)


if __name__ == "__main__":
    main()
