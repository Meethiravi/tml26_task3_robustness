import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18
import torchvision.transforms as T

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 9
EPOCHS      = 60
EPSILON     = 8 / 255.0
ALPHA       = 2 / 255.0

data   = np.load("/home/atml_team060/tml26_task3/train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

# The 4% data boost
n_val = int(0.04 * len(images))
train_ds, val_ds = random_split(TensorDataset(images, labels), [len(images) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# Natural augmentations only (ResNet-18 capacity limit)
augment = nn.Sequential(
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
).to(DEVICE)

model    = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

# Exponential Moving Average (EMA)
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

ema = EMA(model)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

# The MultiStepLR Scheduler (drops at 30 and 45)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

def pgd_attack(x, y, eps=EPSILON, alpha=ALPHA, iters=10):
    was_training = model.training
    model.eval()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    if was_training:
        model.train()
    return x_adv.detach()

def train_epoch():
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_aug = augment(x)
        optimizer.zero_grad()
        
        # 1. Generate adversarial examples
        x_adv = pgd_attack(x_aug, y, iters=7)
        
        # 2. Compute BOTH losses
        loss_clean = criterion(model(x_aug), y)
        loss_adv   = criterion(model(x_adv), y)
        
        # 3. Explicitly optimize for the leaderboard metric (Metric Hacking)
        loss = 0.5 * loss_clean + 0.5 * loss_adv
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
    scheduler.step()

def evaluate(eval_model, loader, attack=False):
    eval_model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack:
            x = pgd_attack(x, y, iters=20) # PGD-20 for realistic evaluation
        with torch.no_grad():
            correct += (eval_model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total

best_score = 0
print(f"{'Epoch':>5} | {'CleanVal':>8} | {'RobVal':>8} | {'Score':>8}")
print("-" * 38)

for epoch in range(1, EPOCHS + 1):
    train_epoch()
    if epoch % 5 == 0 or epoch == EPOCHS:
        clean = evaluate(ema.shadow, val_loader)
        rob   = evaluate(ema.shadow, val_loader, attack=True)
        score = 0.5 * clean + 0.5 * rob
        print(f"{epoch:5d} | {clean:8.3f} | {rob:8.3f} | {score:8.3f}")
        if clean > 0.50 and score > best_score:
            best_score = score
            torch.save(ema.shadow.state_dict(), "model_multisteplr.pt")
            print(f"      -> saved best EMA model ({best_score:.3f})")