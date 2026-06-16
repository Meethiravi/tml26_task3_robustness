# score improved (0.529667)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet50 
import torchvision.transforms as T

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 9
EPOCHS      = 100
FGSM_EPS    = 8 / 255.0

data   = np.load("/home/atml_team060/tml26_task3/train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

n_val = int(0.1 * len(images))
train_ds, val_ds = random_split(TensorDataset(images, labels), [len(images) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

augment = nn.Sequential(T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip()).to(DEVICE)

model    = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def fgsm(x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    nn.CrossEntropyLoss()(model(x_adv), y).backward()
    return (x + eps * x_adv.grad.sign()).clamp(0, 1).detach()

def train_epoch(epoch):
    model.train()
    # Adversarial warmup schedule:
    # Epochs 1-30:  clean only
    # Epochs 31-60: 25% adversarial
    # Epochs 61+:   50% adversarial
    if epoch <= 30:
        adv_ratio = 0.0
    elif epoch <= 60:
        adv_ratio = 0.25
    else:
        adv_ratio = 0.5

    for x, y in train_loader:
        x, y  = x.to(DEVICE), y.to(DEVICE)
        x_aug = augment(x)

        if adv_ratio > 0:
            model.eval()
            x_adv = fgsm(x_aug, y, FGSM_EPS)
            model.train()
            n_adv   = int(adv_ratio * len(x))
            x_mixed = torch.cat([x_aug[:len(x)-n_adv], x_adv[len(x)-n_adv:]])
            idx     = torch.randperm(len(x_mixed), device=DEVICE)
            x_mixed, y_mixed = x_mixed[idx], y[idx]
        else:
            x_mixed, y_mixed = x_aug, y

        optimizer.zero_grad()
        criterion(model(x_mixed), y_mixed).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

def accuracy(loader, attack=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack:
            x = fgsm(x, y, FGSM_EPS)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total

best_score = 0
print(f"{'Epoch':>5} | {'Phase':>10} | {'CleanVal':>8} | {'RobVal':>8} | {'Score':>8}")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    train_epoch(epoch)
    if epoch % 5 == 0 or epoch == EPOCHS:
        phase = "clean" if epoch <= 30 else ("25% adv" if epoch <= 60 else "50% adv")
        clean = accuracy(val_loader)
        rob   = accuracy(val_loader, attack=True)
        score = 0.5 * clean + 0.5 * rob
        print(f"{epoch:5d} | {phase:>10} | {clean:8.3f} | {rob:8.3f} | {score:8.3f}")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "model.pt")
            print(f"      -> saved best model ({best_score:.3f})")

print(f"\nFinal Best Unified Score: {best_score:.4f}")

# Sanity check
check    = resnet50(weights=None)
check.fc = nn.Linear(check.fc.in_features, NUM_CLASSES)
check.load_state_dict(torch.load("model.pt", map_location="cpu"))
check.eval()
with torch.no_grad():
    out = check(torch.randn(1, 3, 32, 32))
print(f"Output shape: {out.shape}  ✓")