import numpy as np
import matplotlib.pyplot as plt

# --- Helper: fix transposed/mismatched grids ---
def align_to_grid(arr, gt):
    if arr.shape[0] != gt.shape[0] and arr.shape[1] == gt.shape[0]:
        arr = arr.T
    if arr.shape[1] == gt.shape[1]:
        return arr
    x_a = np.linspace(0, 1, arr.shape[1])
    x_gt = np.linspace(0, 1, gt.shape[1])
    out = np.zeros((arr.shape[0], gt.shape[1]))
    for i in range(arr.shape[0]):
        out[i] = np.interp(x_gt, x_a, arr[i])
    return out

def neuron_diversity_score(activations):
    flat = activations.reshape(activations.shape[0], -1)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1e-8)
    n = flat.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(flat[i] - flat[j]))
    return float(np.mean(dists)) if dists else float("nan")

def count_params(npz_file):
    return sum(v.size for v in npz_file.values())

# --- Load ground truth ---
gt = np.load("Data/ground_truth.npy")
print("Ground truth shape:", gt.shape)

models = ["cpinn", "qapinn_3q", "qapinn_4q", "qapinn_5q"]
labels = {"cpinn": "cPINN", "qapinn_3q": "QAPINN-3q", "qapinn_4q": "QAPINN-4q", "qapinn_5q": "QAPINN-5q"}
colors = {"cpinn": "black", "qapinn_3q": "blue", "qapinn_4q": "green", "qapinn_5q": "purple"}

preds, residuals, acts, params = {}, {}, {}, {}

for m in models:
    preds[m] = align_to_grid(np.load(f"results/predictions_{m}.npy"), gt)
    residuals[m] = align_to_grid(np.load(f"results/pde_residual_{m}.npy"), gt)
    acts[m] = np.load(f"results/activations_{m}.npz")
    params[m] = count_params(np.load(f"results/weights_{m}.npz"))

# --- Relative L2 errors ---
print("\n--- Relative L2 error vs ground truth ---")
rel_l2 = {}
for m in models:
    rel_l2[m] = np.linalg.norm(preds[m] - gt) / np.linalg.norm(gt)
    print(f"{labels[m]}: {rel_l2[m]:.4f}")

# --- PDE residual error (mean absolute) ---
print("\n--- Mean absolute PDE residual ---")
mean_residual = {}
for m in models:
    mean_residual[m] = float(np.mean(np.abs(residuals[m])))
    print(f"{labels[m]}: {mean_residual[m]:.4f}")

# --- Parameter counts ---
print("\n--- Trainable parameter count ---")
for m in models:
    print(f"{labels[m]}: {params[m]}")

# --- Activation diversity ---
print("\n--- Activation diversity ---")
diversity = {}
for m in models:
    diversity[m] = {}
    for layer_name in acts[m].keys():
        score = neuron_diversity_score(acts[m][layer_name])
        diversity[m][layer_name] = score
        print(f"{labels[m]} {layer_name}: {score:.3f}")

# --- Plot 1: predictions vs ground truth at several time instants ---
time_indices = [0, gt.shape[0]//4, gt.shape[0]//2, gt.shape[0]-1]
fig, axes = plt.subplots(1, len(time_indices), figsize=(20, 4))
for ax, t in zip(axes, time_indices):
    ax.plot(gt[t], label="Ground Truth", color="red", linewidth=2)
    for m in models:
        ax.plot(preds[m][t], label=labels[m], color=colors[m], linestyle="--", alpha=0.8)
    ax.set_title(f"t index = {t}")
    ax.legend(fontsize=7)
plt.suptitle("cPINN vs QAPINN (3q/4q/5q) vs ground truth (Burgers 1D)")
plt.tight_layout()
plt.savefig("results/full_comparison_predictions.png")
plt.show()

# --- Plot 2: bar chart of relative L2 error ---
plt.figure(figsize=(6,4))
plt.bar([labels[m] for m in models], [rel_l2[m] for m in models], color=[colors[m] for m in models])
plt.ylabel("Relative L2 error")
plt.title("Relative L2 error by model")
plt.tight_layout()
plt.savefig("results/bar_l2_error.png")
plt.show()

# --- Plot 3: bar chart of PDE residual ---
plt.figure(figsize=(6,4))
plt.bar([labels[m] for m in models], [mean_residual[m] for m in models], color=[colors[m] for m in models])
plt.ylabel("Mean |PDE residual|")
plt.title("PDE residual error by model")
plt.tight_layout()
plt.savefig("results/bar_pde_residual.png")
plt.show()

# --- Plot 4: QAPINN loss curves comparison (3q/4q/5q); cPINN loss curve not available ---
plt.figure(figsize=(7,5))
for m in ["qapinn_3q", "qapinn_4q", "qapinn_5q"]:
    loss = np.load(f"results/loss_curve_{m}.npy")
    plt.semilogy(loss, label=labels[m], color=colors[m])
plt.xlabel("Epoch")
plt.ylabel("Loss (log)")
plt.title("Training loss: QAPINN 3q vs 4q vs 5q (cPINN loss curve not yet available)")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_comparison_qubits.png")
plt.show()

print("\nDone. All plots saved in the results/ folder.")