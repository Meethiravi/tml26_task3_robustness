import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet50
import torchvision.transforms as T
from tqdm import tqdm

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 9
EPOCHS      = 120
EPSILON     = 0.04
ALPHA       = 2 / 255.0
PGD_ITERS   = 10
PATIENCE    = 15

data   = np.load("/home/atml_team060/tml26_task3/train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

# Using full dataset 
train_loader = DataLoader(TensorDataset(images, labels), batch_size=256,
                          shuffle=True, num_workers=2, pin_memory=True)

# Validation sample
val_size = 5000
val_indices = torch.randperm(len(images))[:val_size]
val_sub_images = images[val_indices]
val_sub_labels = labels[val_indices]

val_sub_loader = DataLoader(
    TensorDataset(val_sub_images, val_sub_labels), 
    batch_size=256, 
    shuffle=False, 
    num_workers=2
)

# Data augmentation
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                             weight_decay=5e-4, nesterov=True)
# Multistep LR scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                             milestones=[60, 90], gamma=0.1)

def pgd_attack(x, y, eps=EPSILON, alpha=ALPHA, iters=PGD_ITERS):
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
    total_loss = correct = total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
    for x, y in pbar:
        x, y  = x.to(DEVICE), y.to(DEVICE)
        x_aug = augment(x)

        x_adv = pgd_attack(x_aug, y)

        x_combined = torch.cat([x_aug, x_adv], dim=0)
        y_combined = torch.cat([y,     y     ], dim=0)

        optimizer.zero_grad()
        logits = model(x_combined)
        loss   = criterion(logits, y_combined)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits[:len(y)].argmax(1) == y).sum().item()
        total      += len(y)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct/total:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.5f}")

    scheduler.step()
    return total_loss / len(train_loader), correct / total

def evaluate(loader, attack=False):
    model.eval()
    correct = total = total_loss = 0
    desc = "Eval (PGD)" if attack else "Eval (clean)"
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack:
            x = pgd_attack(x, y, iters=10)
        with torch.no_grad():
            logits     = model(x)
            loss       = criterion(logits, y)
            correct   += (logits.argmax(1) == y).sum().item()
            total     += len(y)
            total_loss += loss.item()
        pbar.set_postfix(acc=f"{correct/total:.4f}")
    return correct / total, total_loss / len(loader)

# Logs
logs = {
    "epochs":         [],
    "train_loss":     [],
    "train_acc":      [],
    "val_loss_clean": [],
    "val_loss_adv":   [],
    "val_acc_clean":  [],
    "val_acc_adv":    [],
    "score":          [],
}

best_score = 0
best_val_loss = float("inf")
patience_counter = 0

print(f"{'Epoch':>5} | {'LR':>8} | {'TrainLoss':>10} | {'TrainAcc':>9} | {'ValLoss(C)':>10} | {'ValLoss(A)':>10} | {'CleanAcc':>9} | {'RobAcc':>8} | {'Score':>8}")
print("-" * 115)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(epoch)

    if epoch % 5 == 0 or epoch == 1:
        clean_acc, clean_val_loss = evaluate(val_sub_loader, attack=False)
        rob_acc, adv_val_loss = evaluate(val_sub_loader, attack=True)

        score = 0.5 * clean_acc + 0.5 * rob_acc
        lr_now = scheduler.get_last_lr()[0]

        logs["epochs"].append(epoch)
        logs["train_loss"].append(round(train_loss, 6))
        logs["train_acc"].append(round(train_acc, 6))
        logs["val_loss_clean"].append(round(clean_val_loss, 6))
        logs["val_loss_adv"].append(round(adv_val_loss, 6))
        logs["val_acc_clean"].append(round(clean_acc, 6))
        logs["val_acc_adv"].append(round(rob_acc, 6))
        logs["score"].append(round(score, 6))

        print(f"{epoch:5d} | {lr_now:8.5f} | {train_loss:10.4f} | {train_acc:9.4f} | {clean_val_loss:10.4f} | {adv_val_loss:10.4f} | {clean_acc:9.4f} | {rob_acc:8.4f} | {score:8.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "model_pgd.pt")
            print(f"-> saved best model (score={best_score:.4f})")

with open("training_logs.json", "w") as f:
    json.dump(logs, f, indent=2)
print("\nLogs saved -> training_logs.json")

# Sanity check

check    = resnet50(weights=None)
check.fc = nn.Linear(check.fc.in_features, NUM_CLASSES)
check.load_state_dict(torch.load("model_pgd.pt", map_location="cpu"))
check.eval()
with torch.no_grad():
    out = check(torch.randn(1, 3, 32, 32))
print(f"Output shape: {out.shape}")