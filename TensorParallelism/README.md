# QuintNet TensorParallelism

A high-performance PyTorch library for tensor parallelism in distributed deep learning. Designed for seamless integration with existing PyTorch workflows while providing efficient scaling across multiple GPUs.

```
Data Parallelism:     Model Parallelism:      Tensor Parallelism:
┌─────┬─────┬─────┐   ┌─────┐ ┌─────┐        ┌───┬───┬───┬───┐
│  M  │  M  │  M  │   │  L1 │ │  L2 │   →    │ L │ L │ L │ L │
│  o  │  o  │  o  │   │  L2 │ │  L3 │        │ 1 │ 1 │ 1 │ 1 │
│  d  │  d  │  d  │   │  L3 │ │  L4 │        └───┴───┴───┴───┘
│  e  │  e  │  e  │   └─────┘ └─────┘        Split within layer
│  l  │  l  │  l  │   Sequential             Parallel execution
└─────┴─────┴─────┘   execution
Different data        Different layers        Same layer split
```

## ✨ Key Features

* **🚀 One-Line Integration**: Convert any PyTorch model with a single function call via `apply_tensor_parallel`.
* **⚡ Optimized Communication**: Custom autograd communication primitives (All\_Gather, All\_Reduce, ReduceScatter).
* **🔧 Modular Architecture**: Clear separation: comm ops, layers, rewrite utilities, and process group manager.
* **📈 Linear Scaling**: Efficient memory/compute distribution across GPUs.
* **🛡️ Production Ready**: Designed with robust error handling and testability in mind.

---

## 🚀 Quick Start — Conda (recommended for GPU / CUDA users)

> This section explains how to create a conda env, install PyTorch correctly for your CUDA driver, install project dependencies, and run training with `torchrun`.
> **Important:** choose the `pytorch-cuda` package version that matches your system driver (e.g., `11.8` or `12.1`). If unsure, check `nvidia-smi`.

### 1. Create conda env (Python 3.11)

```bash
conda create -n quintnet python=3.11 -y
conda activate quintnet
```

### 2. Install PyTorch with CUDA (examples)

**If you have CUDA 12.1 drivers (example):**

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

**If you have CUDA 11.8 drivers:**

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
```

If you prefer CPU-only:

```bash
conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
```

*(If you’re unsure which CUDA to pick run `nvidia-smi` — check the driver version and choose the best matching `pytorch-cuda` package.)*

### 3. Install other deps & the package (editable)

```bash
# From repo root (/workspace)
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found — skipping"
```

### 4. Verify GPUs & PyTorch

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

---

## 🏁 Running training (single-node / multi-GPU with `torchrun`)

Mnist_Digit_Classification repository contains training scripts at:
**Training repo (GitHub):** https://github.com/Wodlfvllf/Mnist-Digit-Classification

* `Mnist_Digit_Classification/TP_training.py` (tensor-parallel training entry)
* `Mnist_Digit_Classification/train.py` (single-node or module-run compatible)

### 1 GPU (simple)

```bash
python -m Mnist_Digit_Classification.train
# or
python Mnist_Digit_Classification/train.py
```

### 2 GPUs on single node (recommended for your TP code)

Use `torchrun` to spawn one process per GPU:

```bash
# Example: 2 GPUs (single node)
torchrun --standalone --nproc_per_node=2 -m Mnist_Digit_Classification.TP_training
```

### More explicit (master address/port)

```bash
torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  -m Mnist_Digit_Classification.TP_training
```

---

## 🧩 Typical workflow / checklist

1. Create conda env and install PyTorch (match CUDA).
2. Ensure `dist.init_process_group(...)` is called in the entrypoint (TP\_training does this).
3. Run with `torchrun --nproc_per_node=<#GPUs> -m path.to.module`.
4. If you see rank / NCCL warnings, ensure you set `torch.cuda.set_device(local_rank)` in each process and use `nccl` backend on GPUs.

---

## 🏗️ Project structure

```
QuintNet/TensorParallelism/
├── comm_ops.py          # Distributed communication primitives (autograd aware)
├── layers.py            # ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
├── rewrite.py           # apply_tensor_parallel()
├── processgroup.py      # ProcessGroupManager
└── utils.py             # helper functions
```

---

## 📝 Notes & tips (beginner-friendly)

* **Row- vs Column-parallel**: pick the parallelism strategy that fits your model; Column splits input features, Row splits output contributions — both have tradeoffs in communication vs memory.
* **Bias handling**: add bias after gather/reduction to avoid double-counting.
* **Debugging**: print `x.shape` before shards; if you see 2D vs 3D mismatch, unsqueeze or adapt slicing logic (support both `(B,H)` and `(B,T,H)`).
* **Common launch errors**: `No module named ...` → use `-m package.module` from repo root or add repo root to `PYTHONPATH`.
* **NCCL tips**: If multi-node, set `MASTER_ADDR`, `MASTER_PORT`, and ensure networking is open between nodes.

---

## 📚 Further reading & examples
* Read through `DOCUMENT.MD` carefully to see what is being done and get wholesome knowledge.
* See `Mnist_Digit_Classification/TP_training.py` for a worked example using the library.
* Example: run `python -m Mnist_Digit_Classification.TP_training` for single-process debugging; run via `torchrun` for multi-GPU.

---

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines. If adding new parallel layers, add unit tests for shape consistency (2D/3D), and a small torchrun test that runs forward + backward.

---

Thanks for using QuintNet — ping the repo issues if you need help reproducing anything on your machine!
