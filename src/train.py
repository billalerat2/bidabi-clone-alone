"""
Simple image classification training pipeline.

Steps:
1. Load dataset from data/raw/images/<class>/
2. Split into train / val / test (70 / 15 / 15)
3. Apply augmentations (train) and normalization (val/test)
4. Train a ResNet-18 (ImageNet pretrained, full fine-tuning)
5. Track train/val loss and accuracy each epoch
6. Save the best model on validation accuracy

Run:
    python src/train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


# --- Config ---
DATA_DIR = "data/raw/images"
MODEL_PATH = "models/best_resnet18.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-4
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)


# --- 3. Augmentations ---
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- 1. Load dataset (twice, so train and eval get different transforms) ---
base_train = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
base_eval = datasets.ImageFolder(DATA_DIR, transform=eval_tfm)
classes = base_train.classes
print(f"Classes ({len(classes)}): {classes}")
print(f"Total samples: {len(base_train)}")

if len(classes) < 2:
    raise SystemExit(
        f"Need ≥ 2 classes in {DATA_DIR}/. Found {len(classes)}: {classes}. "
        "Run scraping for more categories first."
    )


# --- 2. Split train/val/test (same seed → identical splits on both copies) ---
n = len(base_train)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

train_set, _, _ = random_split(
    base_train, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
_, val_set, test_set = random_split(
    base_eval, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# --- 4. Model: ResNet-18 pretrained, replace final layer ---
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def run_epoch(loader, training):
    """Run one epoch (train or eval). Returns (loss, accuracy)."""
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# --- 5 + 6. Training loop with metric tracking and best-model save ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"val loss={val_loss:.4f} acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {"model": model.state_dict(), "classes": classes},
            MODEL_PATH,
        )
        print(f"  → best model saved ({val_acc:.4f}) → {MODEL_PATH}")


# --- Final test on the best checkpoint ---
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
test_loss, test_acc = run_epoch(test_loader, training=False)

print(f"\nBest val acc: {best_val_acc:.4f}")
print(f"Test acc:     {test_acc:.4f}")
