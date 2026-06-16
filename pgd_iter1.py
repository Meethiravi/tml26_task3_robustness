# score 0.568026

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
EPOCHS      = 120
EPSILON     = 8 / 255.0
ALPHA       = 2 / 255.0
PGD_ITERS   = 7

data   = np.load("/home/atml_team060/tml26_task3/train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

n_val = int(0.1 * len(images))
train_ds, val_ds = random_split(TensorDataset(images, labels), [len(images) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=128,  shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

augment = nn.Sequential(
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
).to(DEVICE)

model    = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def pgd_attack(x, y, eps=EPSILON, alpha=ALPHA, iters=PGD_ITERS):
    # Switch to eval for attack, remember training state
    was_training = model.training
    model.eval()

    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x + delta, 0, 1).detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)

    if was_training:
        model.train()
    return x_adv.detach()

def train_epoch(epoch):
    model.train()
    total_loss = 0

    use_adv = epoch > 40

    for x, y in train_loader:
        x, y  = x.to(DEVICE), y.to(DEVICE)
        x_aug = augment(x)

        optimizer.zero_grad()

        if use_adv:
            x_adv = pgd_attack(x_aug, y)
            loss  = 0.5 * criterion(model(x_aug), y) + \
                    0.5 * criterion(model(x_adv), y)
        else:
            loss = criterion(model(x_aug), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(train_loader)

def evaluate(loader, attack=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack:
            x = pgd_attack(x, y, iters=5)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total

best_score = 0
print(f"{'Epoch':>5} | {'Phase':>10} | {'Loss':>8} | {'CleanVal':>8} | {'RobVal':>8} | {'Score':>8}")
print("-" * 68)

for epoch in range(1, EPOCHS + 1):
    avg_loss = train_epoch(epoch)
    if epoch % 5 == 0 or epoch == 1:
        phase     = "clean" if epoch <= 20 else "50% adv"
        clean     = evaluate(val_loader)
        rob       = evaluate(val_loader, attack=True)
        score     = 0.5 * clean + 0.5 * rob
        print(f"{epoch:5d} | {phase:>10} | {avg_loss:8.3f} | {clean:8.3f} | {rob:8.3f} | {score:8.3f}")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "model_pgd.pt")
            print(f"      -> saved best model ({best_score:.3f})")

print(f"\nFinal Best Unified Score: {best_score:.4f}")

# Sanity check
check    = resnet50(weights=None)
check.fc = nn.Linear(check.fc.in_features, NUM_CLASSES)
check.load_state_dict(torch.load("model_pgd.pt", map_location="cpu"))
check.eval()
with torch.no_grad():
    out = check(torch.randn(1, 3, 32, 32))
print(f"Output shape: {out.shape}  ✓")