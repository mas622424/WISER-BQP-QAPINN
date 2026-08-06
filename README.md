# Quantum-Assisted Physics-Informed Neural Network (QAPINN)
**BQP WISER Quantum Challenge 2026**

## 🎯 The Challenge
The objective of this project is to investigate the explainability and expressivity of a quantum layer within a Physics-Informed Neural Network (PINN). Specifically, we aim to explain when, why, and how the introduction of a Variational Quantum Circuit (VQC) changes the learning dynamics of the network when solving the 1D Viscous Burgers' Equation, a notorious PDE known for developing sharp shockwave discontinuities.

## 🛠️ Our Approach
Rather than just aiming for speed, our methodology focuses on isolating the quantum layer's impact. We developed a pure Classical PINN (cPINN) as a baseline, and compared it against three hybrid variants: 3-qubit, 4-qubit, and 5-qubit QAPINNs. By controlling the expressivity of the quantum layer (via the number of qubits) and comparing the intermediate hidden-layer activations against the classical baseline, we mapped how the quantum entanglement alters the network's capacity to learn physical laws.

## ⚙️ Methods & Tools
* **Quantum/Classical Frameworks:** PyTorch (for the classical neural network stack and autograd PDE calculus) and PennyLane (for simulating the VQC and integrating it as a PyTorch TorchLayer).
* **Environment:** Google Colab (T4 Tensor Core GPU).
* **Data Selection:** Instead of external training data, the models generate internal collocation points dynamically. We benchmarked our final predictions against the standard analytical solution (`burgers_shock.mat` via Raissi et al.) mapped to a 256x100 spatial-temporal grid for zero-interpolation error analysis.

## 📈 Results & Findings
* **Parameter Efficiency:** The integration of the quantum layer resulted in a highly parameter-efficient model. (e.g., The 4-qubit QAPINN utilized only 577 parameters compared to the classical baseline).
* **Learning Dynamics:** The QAPINN successfully minimizes the total physics-informed loss (decreasing from ~0.4 to ~0.1 over 1000 epochs), proving that classical optimizers (Adam) can effectively update quantum circuit parameters in tandem with classical weights using PDE residuals.
* **Neuron Diversity:** [Role 3 to insert brief 1-sentence finding about Layer 2/3 activation diversity here].

## 🔮 Limitations & Recommended Next Steps
* **The Shockwave Convergence Limit:** Despite minimizing the training loss, the basic QAPINN variants struggled to converge on the true analytical solution for the Burgers' equation (e.g., 4-qubit Relative L2 Error ~ 1.20). This highlights a known limitation in standard PINN architectures when resolving sharp shock gradients.
* **Next Steps:** Future development should investigate adaptive collocation point sampling (focusing points around the shockwave), modifying the VQC ansatz to capture higher-frequency Fourier features, or testing the QAPINN on smoother PDEs (like the Heat Equation) to decouple the shockwave difficulty from the quantum layer's native expressivity.

## 📁 Repository Structure
* **/notebooks:** Contains the Google Colab training notebooks for all model variants.
  * `01_cPINN_Baseline.ipynb` - Classical PINN baseline.
  * `02_QAPINN_4_qubit.ipynb` - Hybrid model with a 4-qubit VQC layer.
  * `03_QAPINN_3_qubit.ipynb` - 3-qubit variant (reduced expressivity).
  * `04_QAPINN_5_qubit.ipynb` - 5-qubit variant (increased expressivity).
* **/results:** Contains exported NumPy arrays (`.npy`, `.npz`) used for XAI analysis, including:
  * `predictions_[model].npy`: Predicted fluid velocity u(x,t).
  * `activations_[model].npz`: Captured hidden layer activations.
  * `loss_curve_[model].npy`: Epoch loss tracking.
  * `weights_[model].npz`: Flattened network weights.
  * `ground_truth.npy`: Exact analytical solution for benchmarking.

## 💻 Reproducibility Instructions
To ensure mathematical reproducibility, strict random seeds (`seed = 42`) are enforced across all initializations.
1. Open any `.ipynb` file in Google Colab.
2. Set hardware accelerator to GPU (**Runtime > Change runtime type > T4 GPU**).
3. Run the first cell to install dependencies (`pip install torch pennylane scipy`).
4. Select **Runtime > Run all**. The model will train for 1,000 epochs, dynamically pull the exact analytical data, and save the artifact files to the Colab system.

## 🧠 Neural Network Architecture Details
* **Loss Function:** L_Total = L_PDE + L_IC + L_BC
* **Hybrid VQC Design (PennyLane):**
  * **Data Encoding:** Inputs scaled to [-π, π] via AngleEmbedding (Y-axis rotations).
  * **Trainable Ansatz:** 2-layer heavily entangled circuit applying parameterized rotations (`qml.Rot` via phi, theta, omega angles) followed by ring-topology CNOT gates.
  * **Measurement:** Expectation values of Pauli-Z observables.
* **Classical Stack (PyTorch):** Linear layers utilizing `Tanh` activations to maintain the twice-differentiable requirement for solving second-order PDEs.

## 👥 Team Members & Contributions
* **Dennis Appiah Kubi (Role 1: Theory Lead):** Selected benchmark PDEs, formulated the mathematical derivations, derived the governing equations, and established the VQC mathematical justifications.(GitHub:[Mr-Kad7](https://github.com/Mr-Kad7))
* **Ajay Sankar Makkena (Role 2: ML & Quantum Implementation Lead):** Designed the PyTorch & PennyLane architectures, engineered the automatic differentiation physics engine, executed model training, and generated all experimental data artifacts.(GitHub:[mas622424](https://github.com/mas622424))
* **Pascal Chabo Bya'ombe (Role 3: Explainable AI & Analysis Lead):** Applied XAI techniques to evaluate layer activations, generated comparative performance heatmaps, interpreted model metrics, and compiled the final technical report. (GitHub: [@scal01](https://github.com/scal01))