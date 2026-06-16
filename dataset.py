import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data   = np.load("train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0  # (N, 3, 32, 32)
labels = torch.from_numpy(data["labels"]).long()

N = len(images)
print(f"Total images : {N}")
print(f"Image shape  : {images.shape}")
print(f"Label range  : {labels.min().item()} – {labels.max().item()}")
 
mean = images.mean(dim=[0, 2, 3])
std  = images.std(dim=[0, 2, 3])

print(f"\nPer-channel mean : {mean.tolist()}")
print(f"Per-channel std  : {std.tolist()}")
print("\nPaste these into your training script:")
print(f"MEAN = torch.tensor({[round(v, 4) for v in mean.tolist()]})")
print(f"STD  = torch.tensor({[round(v, 4) for v in std.tolist()]})")

NUM_CLASSES = int(labels.max().item()) + 1
counts = [(labels == c).sum().item() for c in range(NUM_CLASSES)]

print(f"\nClass distribution:")
for c, n in enumerate(counts):
    bar = "█" * (n // 200)
    print(f"  Class {c}: {n:5d}  {bar}")

N_SAMPLES = 8   # columns
fig, axes = plt.subplots(
    NUM_CLASSES, N_SAMPLES,
    figsize=(N_SAMPLES * 1.4, NUM_CLASSES * 1.4)
)
fig.suptitle("Sample images — one row per class", fontsize=12, y=1.01)

for class_idx in range(NUM_CLASSES):
    class_mask    = (labels == class_idx).nonzero(as_tuple=True)[0]
    chosen        = class_mask[torch.randperm(len(class_mask))[:N_SAMPLES]]
    
    for col, img_idx in enumerate(chosen):
        ax  = axes[class_idx][col]
        img = images[img_idx].permute(1, 2, 0).numpy()  # (H, W, 3)
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"Class {class_idx}", fontsize=8,
                          rotation=0, labelpad=30, va="center")

plt.tight_layout()
plt.savefig("dataset_samples.png", dpi=150, bbox_inches="tight")
print("\nSaved image grid → dataset_samples.png")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
channel_names = ["Red", "Green", "Blue"]
colors        = ["red", "green", "blue"]

for ch in range(3):
    ax = axes[ch]
    for class_idx in range(NUM_CLASSES):
        mask   = (labels == class_idx)
        pixels = images[mask, ch].flatten().numpy()
        ax.hist(pixels, bins=50, alpha=0.4, density=True,
                label=f"Class {class_idx}")
    ax.set_title(f"{channel_names[ch]} channel")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")

axes[0].legend(fontsize=6, loc="upper left")
plt.suptitle("Per-channel pixel distributions by class", fontsize=11)
plt.tight_layout()
plt.savefig("dataset_histograms.png", dpi=150, bbox_inches="tight")
print("Saved histograms      → dataset_histograms.png")
plt.show()