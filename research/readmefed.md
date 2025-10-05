# Federated Capsule Network (FedAvg on MedMNIST)

This repository implements **Federated Learning with a Capsule Network (CapsNet)** using the **FedAvg** algorithm on the **OrganMNIST** dataset from [MedMNIST](https://medmnist.com/).  
It preserves the *original Capsule Network architecture* (with dynamic routing, squashing nonlinearity, margin loss, and reconstruction decoder) and extends it to a multi-client federated simulation framework.

---

## 🧠 Overview

Capsule Networks (CapsNets) model hierarchical relationships between features through *capsules* — groups of neurons whose activity vectors represent both the probability and pose of detected entities.  
This project combines the original **Sabour et al. (2017)** CapsNet with **Federated Averaging (FedAvg)** to simulate distributed training across multiple clients while preserving data privacy.

Each client trains locally on its portion of the dataset (either IID or non-IID), and a central server aggregates the model weights weighted by local dataset sizes.

---

## ⚙️ Features

- **CapsNet intact** — full dynamic routing, squashing, and margin + reconstruction loss.  
- **FedAvg implementation** — local client updates and weighted model aggregation.  
- **IID / Non-IID partitioning** — toggle between uniform and label-skewed data using Dirichlet distribution.  
- **MedMNIST integration** — runs on OrganMNIST (`axial`, `coronal`, or `sagittal`).  
- **Device auto-selection** — automatically uses CUDA if available.  
- **Deterministic runs** — consistent results through seeded RNGs.

---

## 🧩 File Structure

├── federated_caps.py # Main training script (CapsNet + FedAvg)
├── data_downloader.py # Helper for choosing MedMNIST dataset class
├── requirements.txt # (optional) Python dependencies
└── README.md # You are here


---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision medmnist numpy
If medmnist fails to install:

pip install git+https://github.com/MedMNIST/MedMNIST.git

2. Dataset location

By default, datasets are saved to:

C:\Users\DELL\medmnist_data


You can change this path inside CONFIG["data_root"] in the script.

3. Run training
python federated_caps.py

🧪 Configuration

Edit the CONFIG dictionary near the top of federated_caps.py to control the experiment.

Key	Description	Default
view	OrganMNIST view ('axial', 'coronal', 'sagittal')	'axial'
num_clients	Number of federated clients	10
frac_clients	Fraction of clients sampled each round	0.5
rounds	Number of global communication rounds	20
local_epochs	Number of local epochs per client	2
iid	Toggle IID (True) or Non-IID (False) partition	True
dirichlet_alpha	Dirichlet α for label-skew (smaller → more skew)	0.5
lr	Learning rate	2e-4
batch_size	Training batch size	128
device	'cuda' or 'cpu' (auto)	auto
seed	Random seed	42
🧮 How It Works

Dataset Partitioning

IID mode: randomly splits the training data equally across clients.

Non-IID mode: simulates label imbalance using a Dirichlet(α) distribution.

Local Training

Each client receives a copy of the global CapsNet.

Clients train locally using the margin + reconstruction loss.

Aggregation (FedAvg)

The server averages client parameters weighted by the number of samples.

Evaluation

After each communication round, the global model is evaluated on the MedMNIST validation set.

🧠 Capsule Network Details

Architecture based on Sabour et al., NeurIPS 2017:

Layer	Description
Conv Layer	256 filters, 9×9 kernel, ReLU
PrimaryCaps	8 capsule types × 32 convs, stride 2
DigitCaps	11 capsules × 16 D vectors, 3 routing iterations
Decoder	3-layer MLP (512 → 1024 → 784)
Loss	
𝐿
=
𝐿
𝑚
𝑎
𝑟
𝑔
𝑖
𝑛
+
0.0005
×
𝐿
𝑟
𝑒
𝑐
𝑜
𝑛
𝑠
𝑡
𝑟
𝑢
𝑐
𝑡
𝑖
𝑜
𝑛
L=L
margin
	​

+0.0005×L
reconstruction
	​

🧬 Example Experiments
IID Federated Training
CONFIG["iid"] = True
CONFIG["rounds"] = 15
CONFIG["num_clients"] = 5
python federated_caps.py

Non-IID (Label-Skewed) Training
CONFIG["iid"] = False
CONFIG["dirichlet_alpha"] = 0.3
python federated_caps.py


Compare accuracy curves between IID and Non-IID setups to observe how heterogeneity affects convergence.

📊 Expected Output

Console logs per round:

Round 01 | Clients: 5/10 | Val Acc: 0.7421 | Time: 33.4s
Round 02 | Clients: 5/10 | Val Acc: 0.7812 | Time: 34.0s
...
Training finished in 612.3s
Best Validation Accuracy: 0.8125
Final Test Accuracy: 0.7962

🧪 Key References

Capsule Network:
Sabour, S., Frosst, N., & Hinton, G. E. (2017).
Dynamic Routing Between Capsules. NeurIPS.

Federated Averaging:
McMahan, B. et al. (2017).
Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.

Dataset:
Yang, J., Shi, R., Ni, B. et al. (2021).
MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data 8, 245.

📈 Visualization (optional)

To visualize convergence between IID and Non-IID runs:

import matplotlib.pyplot as plt
rounds = list(range(len(acc_iid)))
plt.plot(rounds, acc_iid, label="IID")
plt.plot(rounds, acc_noniid, label="Non-IID")
plt.xlabel("Communication Rounds")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.title("FedAvg Convergence of CapsNet")
plt.show()