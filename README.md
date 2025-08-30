# QuintNet — Towards 5D Parallelism for Scalable Deep Learning

QuintNet is a research-oriented PyTorch framework designed to explore and implement multi-dimensional parallelism strategies for distributed deep learning. Our goal is to enable efficient training and inference of massive foundation models that exceed the capacity of single GPUs or even single nodes. By combining multiple parallelism dimensions—often referred to as "5D Parallelism"—QuintNet provides a modular, extensible toolkit for scaling deep learning workflows across clusters.

We start with Tensor Parallelism as the foundational module and plan to progressively integrate other strategies like Pipeline, Sequence, Expert/MoE, and Data Parallelism. This allows users to experiment with hybrid approaches tailored to their models and hardware.

![QuintNet Logo or Banner](https://github.com/Wodlfvllf/QuintNet/blob/main/imgs/Quintnet.jpeg) <!-- Replace with actual logo if available -->

## 🔥 Motivation: Why 5D Parallelism?

Modern AI models (e.g., large language models like GPT or vision transformers) have billions or trillions of parameters, demanding enormous memory and compute resources. Training them efficiently requires distributing the workload across multiple devices. Traditional single-strategy parallelism often hits bottlenecks:

- **Data Parallelism (DP)**: Replicates the model across devices, each processing different data batches. Great for throughput but memory-intensive.
- **Model Parallelism (MP)**: Splits the model layers across devices. Reduces memory per device but increases communication latency.
- **Tensor Parallelism (TP)**: Shards individual layers (e.g., linear weights) across devices. Enables intra-layer parallelism for better utilization.
- **Pipeline Parallelism (PP)**: Divides the model into stages, pipelining mini-batches through them. Balances load but can suffer from bubble inefficiencies.
- **Expert/MoE Parallelism**: Routes inputs to sparse "experts" (sub-models), scaling capacity without full activation.

**5D Parallelism** combines these (DP + MP + TP + PP + MoE) for optimal scaling. QuintNet aims to make this accessible in PyTorch, inspired by frameworks like DeepSpeed and Megatron-LM, but with a focus on simplicity, modularity, and educational value.

Visualizing the strategies:

```
Data Parallelism:     Model Parallelism:      Tensor Parallelism:     Pipeline Parallelism:   Expert/MoE Parallelism:
┌─────┬─────┬─────┐   ┌─────┐ ┌─────┐         ┌───┬───┬───┬───┐       ┌───┬───┐ ┌───┬───┐      ┌───┐ ┌───┐ ┌───┐
│  M  │  M  │  M  │   │  L1 │ │  L2 │   →     │ L │ L │ L │ L │  →    │ S │ S │ │ S │ S │  →   │ E │ │ E │ │ E │
│  o  │  o  │  o  │   │  L2 │ │  L3 │         │ 1 │ 1 │ 1 │ 1 │       │ t │ t │ │ t │ t │      │ x │ │ x │ │ x │
│  d  │  d  │  d  │   │  L3 │ │  L4 │         └───┴───┴───┴───┘       └───┴───┘ └───┴───┘      └───┘ └───┘ └───┘
│  e  │  e  │  e  │   └─────┘ └─────┘         Split inside layer      Split by pipeline      Experts/Mixture routing
└─────┴─────┴─────┘   Sequential execution    Parallel execution      Stage execution        Conditional parallelism
Different data        Different layers        Same layer split        Sequential stages      Sparse activation
```

QuintNet's hybrid approach allows mixing these for scenarios like:
- TP + PP for large transformers.
- MoE + DP for sparse, high-capacity models.
- Full 5D for extreme-scale training (e.g., on 1000+ GPUs).

## ✨ Key Features & Highlights

- **Modular Design**: Each parallelism strategy is a self-contained module, easy to mix and match.
- **PyTorch-Native**: Builds on `torch.distributed` with custom autograd-aware ops for minimal overhead.
- **Scalability**: Linear scaling in memory and compute; supports single-node multi-GPU and multi-node clusters.
- **Educational Focus**: Clean code with comments, examples, and docs for learning distributed DL.
- **Implemented Modules**:
  - **Tensor Parallelism (✅ Done)**: Efficient sharding of linear layers (column/row parallel) with optimized comm primitives (All_Gather, All_Reduce, ReduceScatter). See [TensorParallelism/README.md](QuintNet/TensorParallelism/README.md) for details.
- **Work-in-Progress (WIP)**:
  - Pipeline Parallelism: Layer partitioning with micro-batching.
- **Planned**:
  - Sequence Parallelism: Sharding along sequence dims for long-context models.
  - Expert/MoE Parallelism: Token routing to experts with load balancing.
  - Data Parallelism Enhancements: Integration with DDP/ZeRO for hybrid setups.
  - Orchestrator: Auto-config for combining strategies based on model/hardware.

## 📂 Repository Structure

```
QuintNet/
├── TensorParallelism/          # Tensor parallelism module
│   ├── comm_ops.py             # Autograd-aware communication primitives
│   ├── layers.py               # Parallel layers (e.g., ColumnParallelLinear)
│   ├── rewrite.py              # Model rewriting utilities
│   ├── processgroup.py         # Process group management
│   ├── utils.py                # Helpers
│   └── README.md               # Submodule docs
├── Mnist_Digit_Classification/ # Example training scripts
│   ├── TP_training.py          # Tensor-parallel entrypoint
│   ├── train.py                # Single-device training
│   └── ...                     # Models, datasets
├── Dataset/                    # Sample datasets (e.g., MNIST)
├── test.py                     # Quick tests
├── DOCUMENT.MD                 # In-depth technical notes
├── CONTRIBUTING.md             # Contribution guidelines
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup (Conda Recommended for GPU/CUDA)

Create a fresh environment:

```bash
conda create -n quintnet python=3.11 -y
conda activate quintnet
```

Install PyTorch (match your CUDA version—check with `nvidia-smi`):

For CUDA 12.1:
```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

For CUDA 11.8:
```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
```

CPU-only:
```bash
conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python - <<'PY'
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY
```

### 3. Clone and Run Examples

The training examples are in a companion repo: [Mnist-Digit-Classification](https://github.com/Wodlfvllf/Mnist-Digit-Classification).

Single GPU/CPU:
```bash
python -m Mnist_Digit_Classification.train
```

Multi-GPU (Tensor Parallelism, e.g., 2 GPUs):
```bash
torchrun --standalone --nproc_per_node=2 -m Mnist_Digit_Classification.TP_training
```

For explicit control (multi-node ready):
```bash
torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  -m Mnist_Digit_Classification.TP_training
```

## 🧩 Typical Workflow

1. **Setup Env**: Follow Quick Start.
2. **Init Distributed**: Use `dist.init_process_group(backend='nccl')` in your script.
3. **Apply Parallelism**: `from QuintNet.TensorParallelism import apply_tensor_parallel; model = apply_tensor_parallel(model, tp_size=world_size)`.
4. **Train**: Launch with `torchrun` for multi-process spawning.
5. **Debug**: Set `local_rank` via env vars; log per-rank with `print(f"[Rank {rank}] ...")`.

## 📝 Notes & Tips

- **Choosing Strategies**: Start with Tensor Parallelism for intra-layer scaling. Combine with Pipeline for deep models.
- **Debugging Distributed Code**: Use barriers (`dist.barrier()`) and assert shapes/values across ranks.
- **Common Pitfalls**: Ensure inputs are replicated for TP; handle biases post-comm to avoid desync.
- **Performance**: Monitor with `torch.profiler` for comm bottlenecks.
- **Further Reading**: Check `DOCUMENT.MD` for technical deep-dive.

## 🏗️ Roadmap

- [x] Tensor Parallelism (Column/Row sharding with custom ops)
- [ ] Pipeline Parallelism (Micro-batching and scheduling)
- [ ] Sequence Parallelism (Long-sequence handling)
- [ ] Expert/MoE Parallelism (Routing and sparsity)
- [ ] Full 5D Orchestrator (Auto-hybrid configs)
- [ ] Benchmarks on large models (e.g., GPT-2 variants)
- [ ] Multi-node support with SLURM integration

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Focus areas: New parallel modules, unit tests, examples, or docs. Please add tests for forward/backward consistency.

## 🏆 Acknowledgments

Inspired by:
- PyTorch Distributed
- DeepSpeed (Microsoft)
- Megatron-LM (NVIDIA)
- FairScale/FairSeq (Meta)

QuintNet is built for experimentation—feel free to open issues or PRs!

⚡ From Tensor Parallelism to Full 5D: Scaling AI Together.
