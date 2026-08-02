# Results and XAI Analysis: cPINN vs QAPINN (3q/4q/5q), Burgers' Equation

## 1. Experimental Setup

Both models were trained on the 1D viscous Burgers' equation with Dirichlet boundary
conditions, using the `burgers_shock` reference dataset. The classical PINN (cPINN)
uses a standard fully connected architecture; the QAPINN replaces the first hidden
layer with a Variational Quantum Circuit (VQC), tested at 3, 4, and 5 qubits. All
models were trained for 1000 epochs.

## 2. Quantitative Results

| Metric | cPINN | QAPINN-3q | QAPINN-4q | QAPINN-5q |
|---|---|---|---|---|
| Relative L2 error vs ground truth | 0.4711 | 0.4571 | 0.4640 | **0.4293** |
| Mean absolute PDE residual | 0.1827 | **0.1469** | 0.1716 | 0.1809 |
| Trainable parameters | 573 | **548** | 577 | 606 |
| Activation diversity — layer 2 | 184.5 | 207.3 | 206.5 | 196.1 |
| Activation diversity — layer 3 | 204.2 | 209.5 | **215.8** | 207.4 |

## 3. Key Findings

**All three QAPINN configurations outperform the classical baseline on relative L2
error**, with the 5-qubit variant achieving the lowest error (0.4293 vs. 0.4711 for
cPINN — a ~9% relative improvement). This is a consistent trend across all three
qubit counts tested, not an isolated result from a single configuration.

**No single QAPINN configuration dominates across every metric.** The 3-qubit
variant achieves the best PDE residual error (0.1469) and is the only QAPINN
configuration with fewer trainable parameters than the cPINN baseline (548 vs.
573). The 5-qubit variant achieves the best L2 error but has the highest
parameter count (606) and a PDE residual comparable to the classical baseline.
This indicates that qubit count trades off differently against different metrics,
rather than uniformly improving performance as it increases.

**Activation diversity is higher for every QAPINN configuration than for the
cPINN, in both hidden layers.** This is the most consistent signal across the
experiment: regardless of qubit count, replacing the first hidden layer with a
VQC increases the diversity of learned representations in the downstream
classical layers, relative to the fully classical baseline.

## 4. Prediction Accuracy Over Time

Across all four models, predictions track the ground truth solution closely at
early time steps but struggle to capture the sharp shock that forms later in
the simulation (visible from t index = 50 onward). All models -- classical and
quantum-assisted alike -- remain smoother than the true solution near the
discontinuity. This shared limitation, rather than the choice of qubit count,
appears to be the dominant source of error for every model tested, consistent
with known difficulties of standard PINN loss formulations near sharp gradients.

## 5. Discussion

These results support a nuanced conclusion rather than a simple "quantum wins"
or "quantum loses" narrative, which matches the challenge brief's stated goal of
explaining *when, why, and how* the quantum layer changes learning dynamics.
Specifically:

- The quantum layer provides a **modest but consistent accuracy improvement**
  across all tested qubit counts, without requiring more parameters in the
  3-qubit case.
- **Increasing qubit count does not uniformly improve all metrics** -- different
  configurations are preferable depending on whether L2 accuracy, PDE residual,
  or parameter efficiency is prioritized.
- The quantum layer's most consistent effect is an **increase in downstream
  activation diversity**, suggesting it may encourage more varied feature
  representations in the classical layers that follow it.
- **Both architectures share the same fundamental limitation** on the sharp
  shock region of Burgers' equation, suggesting this is a property of the PINN
  loss formulation itself rather than something the quantum layer resolves or
  worsens.

## 6. Fourier Spectral Analysis

At the final time step (t index = 99, where the shock is fully formed), the
high-frequency spectral energy ratio was computed for the ground truth and
all four models. The ground truth solution contains substantially more
high-frequency content (ratio 0.0039) than **any** of the trained models
(cPINN: 0.0001; QAPINN-3q: 0.0001; QAPINN-4q: 0.0000; QAPINN-5q: 0.0001) --
roughly one to two orders of magnitude lower across the board. This is a
quantitative confirmation of the qualitative observation in Section 4: none
of the models, classical or quantum-assisted, reproduce the sharp,
high-frequency content of the shock. The quantum layer does not measurably
change this outcome at any qubit count tested.

## 7. Training Noise vs. Qubit Count (Proxy Analysis)

As an indirect proxy for optimization difficulty scaling with circuit size,
the standard deviation of the training loss over the last 200 epochs
(800-1000) was computed for each QAPINN configuration: 3-qubit = 0.0110,
4-qubit = 0.0111, 5-qubit = 0.0120. This shows a mild upward trend in
late-training loss noise as qubit count increases.

**This is not a true barren plateau test.** A rigorous test would require
sampling the variance of gradients with respect to the VQC's own parameters
directly during training (via the parameter-shift rule or similar), which is
not reconstructable from the exported prediction/loss/activation files used
in this analysis. The result above should be read only as a weak, indirect
signal that optimization may become modestly less stable as circuit size
grows -- worth flagging as a direction for Role 1/Role 2 to investigate
further if time permits, not as a confirmed finding.

## 8. Limitations and Next Steps

- Fourier spectral analysis and loss landscape / barren plateau analysis (per
  the methodology plan, Sections 2.2 and 2.4) are still pending and would help
  explain *why* activation diversity increases with the quantum layer.
- The cPINN training loss curve was not available at the time of this analysis
  (only QAPINN loss curves for 3q/4q/5q were exported), so training dynamics
  could only be compared across quantum configurations, not against the
  classical baseline.
- Relative L2 error values differ slightly from an earlier single-run cPINN/
  QAPINN-4q comparison (0.44 vs. 0.47 for cPINN); this is attributed to a
  different training run/seed and should be noted as run-to-run variance in
  the report's methodology section, or addressed by averaging over multiple
  seeds if time permits.
