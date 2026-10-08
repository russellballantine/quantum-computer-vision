# Quantum Computer Vision: Hybrid QCNN

An end-to-end, hardware-accelerated **Hybrid Quantum-Classical Convolutional Neural Network (QCNN)** built using **PennyLane** and **PyTorch**. The network processes downsampled structural images from the MNIST dataset by running a parameterized variational quantum circuit (VQC) as a local feature extraction filter over a classical sliding spatial window.

---

## System Architecture & Data Flow

The network distributes its execution across an isolated hybrid pipeline, balancing state-vector simulations with local hardware registers:

1. Input Handling: A 14x14 normalized image is processed on CPU RAM.
2. Spatial Patching: A sliding 2x2 window (stride=2) extracts localized pixel features.
3. Quantum Filter: Features are fed into a 4-qubit circuit (Angle Encoding -> Parameterized Rotations -> Entanglement Chain).
4. Expectation Readout: Pauli-Z measurements produce a classical 4-channel, 7x7 spatial feature map.
5. Hardware Bridge: Tensors are transferred from CPU RAM directly into Apple Silicon GPU registers using `.to(device)`.
6. Classical Deep Learning: An nn.Conv2d block expands the channels, followed by Adaptive Average Pooling and ReLU activations.
7. Optimization: A final Fully Connected layer outputs class logits. Backpropagation via the Adam optimizer updates both classical weights and quantum gate parameters simultaneously.

---

## Mathematical Formalism

### 1. Spatial Patch Extraction & Feature Map Pre-Processing
An input image is classically resized using bilinear interpolation down to a 14x14 grid to match the simulation bounds of current near-term hardware profiles. A sliding spatial window of size 2x2 extracts localized sub-matrices with a stride of 2. Each local patch is flattened into a real-valued feature vector:
x = [x_0, x_1, x_2, x_3]^T in R^4

### 2. State Preparation via Parametric Angle Encoding
To map classical real numbers into the quantum state space, the vector elements parameterize a series of parallel single-qubit rotations. Given a ground computational base state |0⟩, the state preparation unitary applies a rotation around the y-axis of the Bloch sphere for each target qubit:

|ψ(x)⟩ = Ry(x_0)|0⟩ ⊗ Ry(x_1)|0⟩ ⊗ Ry(x_2)|0⟩ ⊗ Ry(x_3)|0⟩

The rotation matrix generator is defined as:
Ry(x_i) = [[cos(x_i/2), -sin(x_i/2)], [sin(x_i/2), cos(x_i/2)]]

### 3. Variational Circuit Processing & Entanglement
The initialized state vector evolves through a parameterizable ansatz structured over consecutive layers. Each layer applies independent single-qubit rotations followed by a closed linear chain of controlled-NOT (CNOT) operations to maximize quantum entanglement across spatial features. The entanglement wraps circularly around the register:
CNOT(0->1), CNOT(1->2), CNOT(2->3), CNOT(3->0)

### 4. Quantum Feature Expectation Readout
The processed quantum state vector collapses into classical scalar observables by calculating the expectation value of the Pauli-Z operator on each individual wire. For a given patch at a spatial coordinate, the quantum filter yields a 4-dimensional vector of real values bounded between -1 and 1:
y_i = ⟨ψ(x)| U^† Z_i U |ψ(x)⟩  for each qubit i

Because the spatial sliding loop passes over a 14x14 grid with a stride of 2, the final reconstructed output feature map transitions into a classical tensor shape of (batch_size, 4, 7, 7) before hitting the classical network layers.

---
## Circuit Design

```text
q_0: ──RY(x_0)───RX(θ_0)───RY(θ_4)────■───────────────■───────⟨Z_0⟩──
                                      │               │
q_1: ──RY(x_1)───RX(θ_1)───RY(θ_5)────X───────■───────│───────⟨Z_1⟩──
                                              │       │
q_2: ──RY(x_2)───RX(θ_2)───RY(θ_6)────────────X───────│───■───⟨Z_2⟩──
                                                      │   │
q_3: ──RY(x_3)───RX(θ_3)───RY(θ_7)────────────────────X───X───⟨Z_3⟩──

[ Encoding ] [    Parametric Rotations    ] [ Entanglement ] [ Readout ]

```


## Local Hardware Setup & Environment

This repository utilizes isolated project configurations via Python virtual environments and takes advantage of native hardware acceleration for classical processing steps.

### 1. Requirements
* Python 3.9+
* macOS with Apple Silicon (M1/M2/M3/M4 Core Series GPU) or standard Linux/Windows environments.

### 2. Environment Installation
Clone the repository and spin up your local environment bubble:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verification Pipeline Execution
Confirm your math transformations and hardware bridges run correctly by invoking the system core scripts independently in your terminal:
```bash
# Verify the 14x14 downsampling pipeline
python3 src/dataset.py

# Check the PennyLane 4-qubit state-vector circuit loop
python3 src/quantum_filter.py

# Test classical GPU tensor conversions and forward passes
python3 src/model.py

# Fire up the unified hybrid MPS optimization test loop
python3 src/train.py
```

---

## Project Evolution & Future Roadmap

- [x] Implement localized sliding-window 2x2 spatial patch extraction.
- [x] Configure PyTorch-MPS memory bridge for accelerated classical CNN backpropagation loops.
- [ ] Refactor state preparation away from linear single-qubit Angle Encoding to a Logarithmic Amplitude Encoding mechanism where features are mapped directly to state coefficients.
- [ ] Design and build a comparative benchmarking module to directly evaluate classical convergence rates, parameter efficiency, and structural accuracy losses between the 1-to-1 Angle mapping approach and the high-density Logarithmic Amplitude representation framework.
