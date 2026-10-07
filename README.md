# Hybrid Quantum-Classical Convolutional Neural Networks (QCNN) for Computer Vision

## 📌 Project Overview
This repository implements a **Hybrid Quantum-Classical Convolutional Neural Network (QCNN)** designed to evaluate the performance, parameter efficiency, and structural dynamics of Variational Quantum Circuits (VQCs) within computer vision architectures. 

Drawing directly upon core paradigms studied during my **MSc in Data Science at the University of Edinburgh** (specifically within the *Computer Vision* and *Advanced Vision* modules), this project translates classical spatial feature extraction kernels into quantum-enhanced state representations. By implementing a hybrid pipeline using **PennyLane** and **PyTorch**, this project benchmarks how quantum variational layers optimize spatial classification tasks using significantly fewer parameters than traditional classical operations.

---

## 🔬 Architectural Design
The architecture bridges classical tensor pipelines with quantum hardware simulators via a "quanvolutional" approach:

1. **Classical Image Preprocessing:** Images are loaded, normalized, and downsampled to structurally dense patches using standard PyTorch transforms.
2. **Quantum Feature Extraction Layer:** A sliding 2×2 spatial window extracts localized pixel intensities, which are mapped into rotation angles (RY gates) across a 4-qubit register.
3. **Variational Circuit Processing:** A parametric trainable quantum circuit layer applies entangling operations (CNOT) and parameterized rotations (RX), optimized continuously via automatic differentiation.
4. **Classical Feed-Forward Tail:** Quantum expectation values (\(\langle Z \rangle\)) are extracted and routed into a classical neural network for final multi-class projection and cross-entropy evaluation.

---

## 📂 Repository Structure
```text
quantum-computer-vision/
├── data/                    # Local directory for cached datasets (e.g., MNIST/FashionMNIST)
├── notebooks/               # Experimental prototyping and visualization logs
├── src/                     # Core system implementation modules
│   ├── __init__.py
│   ├── dataset.py           # PyTorch data handling and downsampling pipelines
│   ├── model.py             # Hybrid QCNN network definitions and PennyLane wrappers
│   └── train.py             # Optimization loops, loss tracking, and metrics logging
├── .gitignore               # Strict exclusion constraints for security and cache optimization
├── README.md                # Project documentation and architectural manifest
└── requirements.txt         # Explicitly pinned library dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd quantum-computer-vision
   ```

2. **Environment Configuration:**
   It is highly recommended to use a virtual environment (`venv` or `conda`) to manage dependencies securely.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Future Milestones & Deployment Strategy
* **Qiskit Native Backend Execution:** Incorporate the `pennylane-qiskit` plugin to compile optimization circuits directly into Qiskit Runtime Primitives (Sampler/Estimator v2) for testing execution on simulated noisy physical devices.
* **Complex Spatial Benchmarking:** Advance from standard binary classification tasks toward structured edge-detection operations to measure feature-map preservation on complex image boundaries.

---

## 📚 References
* [1] *Systematic Review of Quantum Deep Learning Models for Image Classification.* (Recent comprehensive assessment tracking QCNN, QViT, and Attention frameworks).
* [2] *Systematic Literature Review of Quantum Convolutional Neural Networks.* (Analyzing the design choices of quantum kernels vs classical filters).
* [3] *QCQ-CNN and QCHNet Structural Analyses.* (Demonstrating spatial extraction capabilities and parameter-saving advantages of hybrid architectures under hardware constraints).
