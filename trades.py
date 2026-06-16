import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18
import torchvision.transforms as T

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 9
EPOCHS      = 100
EPSILON     = 8 / 255.0
BETA        = 6.0

# ── Data Loading (10% Validation Split) ───────────────────────────────────────
data   = np.load("/home/atml_team060/tml26_task3/train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

n_val = int(0.1 * len(images))
train_ds, val_ds = random_split(TensorDataset(images, labels), [len(images) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# ── Augmentations & Architecture ──────────────────────────────────────────────
augment = nn.Sequential(
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
).to(DEVICE)

model    = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

# ── Exponential Moving Average (EMA) ──────────────────────────────────────────
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

# Continuous Cosine Schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── TRADES Objective ──────────────────────────────────────────────────────────
def trades_loss(model, x, y, step_size=0.007, perturb_steps=10, beta=BETA):
    # 1. Clean loss
    logits = model(x)
    loss_natural = F.cross_entropy(logits, y)
    
    # 2. Generate adversarial examples specifically for TRADES (maximizing KL Divergence)
    model.eval()
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).to(DEVICE).detach()
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x), dim=1),
                               reduction='sum')
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - EPSILON), x + EPSILON)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    # 3. Calculate final TRADES loss
    model.train()
    loss_robust = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                           F.softmax(model(x), dim=1),
                           reduction='batchmean')
    
    return loss_natural + beta * loss_robust

# ── Training & Evaluation Loops ───────────────────────────────────────────────
def train_epoch():
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_aug = augment(x)
        
        optimizer.zero_grad()
        loss = trades_loss(model, x_aug, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        
    scheduler.step()

def pgd_eval_attack(eval_model, x, y, eps=EPSILON, alpha=2/255.0, iters=20):
    eval_model.eval()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(eval_model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()

def evaluate(eval_model, loader, attack=False):
    eval_model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack:
            x = pgd_eval_attack(eval_model, x, y)
        with torch.no_grad():
            correct += (eval_model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total

# ── Main Loop ─────────────────────────────────────────────────────────────────
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
            torch.save(ema.shadow.state_dict(), "model_trades.pt")
            print(f"      -> saved best TRADES model ({best_score:.3f})")