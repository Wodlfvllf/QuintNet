<p align="center">
  <h1 align="center">🚀 QuintNet</h1>
  <p align="center"><strong>A PyTorch Framework for 3D Distributed Deep Learning</strong></p>
  <p align="center">
    <em>Data Parallel • Pipeline Parallel • Tensor Parallel</em>
  </p>
</p>

---

## ✨ Overview

QuintNet is an educational and production-ready PyTorch library that implements **3D parallelism** for training large-scale deep learning models across multiple GPUs. It provides clean, well-documented implementations of:

- **Data Parallelism (DP)** - Replicate model, split data
- **Pipeline Parallelism (PP)** - Split model layers across GPUs  
- **Tensor Parallelism (TP)** - Split individual layers across GPUs
- **Hybrid 3D Parallelism** - Combine all three for maximum scalability

```
┌─────────────────────────────────────────────────────────────┐
│                     3D Parallelism                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Data       │  │  Pipeline   │  │  Tensor     │         │
│  │  Parallel   │──│  Parallel   │──│  Parallel   │         │
│  │  (Batch)    │  │  (Layers)   │  │  (Weights)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Modular Design** | Each parallelism strategy is independent and composable |
| **1F1B Schedule** | Efficient pipeline schedule minimizing memory footprint |
| **Gradient Bucketing** | Optimized gradient synchronization for DP |
| **Device Mesh** | Flexible N-dimensional device topology |
| **Zero Boilerplate** | Simple strategy-based API for applying parallelism |

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NCCL backend for distributed training

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/QuintNet.git
cd QuintNet

# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### PyTorch with CUDA (Recommended)
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 🚀 Quick Start

### Training with 3D Parallelism

```python
from QuintNet import Trainer, get_strategy, init_process_groups

# Initialize distributed environment
pg_manager = init_process_groups(
    mesh_dim=[2, 2, 2],           # [DP, TP, PP] dimensions
    mesh_name=['dp', 'tp', 'pp']
)

# Apply 3D parallelism strategy
strategy = get_strategy('3d', pg_manager, config)
parallel_model = strategy.apply(model)

# Train with the Trainer
trainer = Trainer(parallel_model, train_loader, val_loader, config, pg_manager)
trainer.fit()
```

### Running Examples

```bash
# Single-node, 8 GPUs with 3D parallelism
torchrun --nproc_per_node=8 -m QuintNet.examples.full_3d --config QuintNet/examples/config.yaml

# Or using Modal for cloud training
modal run train_modal_run.py
```

## 📁 Project Structure

```
QuintNet/
├── core/                      # Core distributed primitives
│   ├── communication.py       # Send, Recv, AllGather, AllReduce
│   ├── device_mesh.py         # N-dimensional device topology
│   └── process_groups.py      # Process group management
│
├── parallelism/
│   ├── data_parallel/         # Data Parallelism (DDP)
│   │   ├── core/ddp.py        # DataParallel wrapper
│   │   └── components/        # Gradient reducer, parameter broadcaster
│   │
│   ├── pipeline_parallel/     # Pipeline Parallelism
│   │   ├── wrapper.py         # PipelineParallelWrapper
│   │   ├── schedule.py        # 1F1B and AFAB schedules
│   │   └── trainer.py         # PipelineTrainer
│   │
│   └── tensor_parallel/       # Tensor Parallelism
│       ├── layers.py          # ColumnParallelLinear, RowParallelLinear
│       └── model_wrapper.py   # Automatic layer replacement
│
├── coordinators/              # Multi-strategy coordinators
│   └── hybrid_3d_coordinator.py
│
├── strategy/                  # High-level strategy API
│   ├── base.py
│   └── strategies/            # DP, PP, TP, 3D strategies
│
├── trainer.py                 # Main Trainer class
│
├── docs/
│   └── TRAINING_GUIDE.md      # 📖 Complete training workflow guide
│
└── examples/
    ├── full_3d.py             # Complete 3D training example
    ├── simple_dp.py           # Data Parallel example
    ├── simple_pp.py           # Pipeline Parallel example
    ├── simple_tp.py           # Tensor Parallel example
    └── config.yaml            # Training configuration
```

## ⚙️ Configuration

Create a `config.yaml` file:

```yaml
# Training
dataset_path: /path/to/dataset
batch_size: 32
num_epochs: 10
learning_rate: 1e-4
grad_acc_steps: 2

# Model
img_size: 28
patch_size: 4
hidden_dim: 64
depth: 8
n_heads: 4

# Parallelism
mesh_dim: [2, 2, 2]        # [DP, TP, PP]
mesh_name: ['dp', 'tp', 'pp']
strategy_name: '3d'
schedule: '1f1b'
```

## 🔧 Parallelism Strategies

### Data Parallelism
Replicates the full model on each GPU. Each GPU processes a different batch, gradients are synchronized via AllReduce.

```bash
torchrun --nproc_per_node=4 -m QuintNet.examples.simple_dp
```

### Pipeline Parallelism
Splits model layers across GPUs. Uses micro-batching with 1F1B schedule for efficiency.

```bash
torchrun --nproc_per_node=4 -m QuintNet.examples.simple_pp
```

### Tensor Parallelism  
Splits individual layer weights across GPUs. Useful for very large layers (e.g., LLM attention/FFN).

```bash
torchrun --nproc_per_node=2 -m QuintNet.examples.simple_tp
```

### 3D Hybrid Parallelism
Combines all three strategies. Requires `DP × TP × PP` GPUs.

```bash
# 8 GPUs: 2 DP × 2 TP × 2 PP
torchrun --nproc_per_node=8 -m QuintNet.examples.full_3d
```

## 📊 Results

Training a Vision Transformer on MNIST with 8 GPUs (2×2×2 mesh):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 1.3817 | 50.46% | 0.8921 | 69.30% |
| 2 | 0.6662 | 77.72% | 0.5135 | 84.52% |
| 3 | 0.4219 | 86.33% | 0.3477 | 89.24% |
| 4 | 0.3214 | 90.02% | 0.2883 | 91.16% |
| 5 | 0.2728 | 91.86% | 0.2509 | 92.06% |
| 6 | 0.2477 | 92.96% | 0.2510 | 92.50% |
| 7 | 0.2364 | 93.78% | 0.2464 | 92.76% |
| 8 | 0.2355 | 94.36% | 0.2372 | 93.18% |
| 9 | 0.2450 | 94.46% | 0.2726 | 93.16% |
| 10 | 0.2573 | 94.80% | 0.3190 | **93.24%** |

**Final Accuracy: 93.24% | Training Time: 1120.72 seconds (~18.7 minutes)**

### Training Configuration
- **Model**: Vision Transformer (64 hidden dim, 8 blocks, 4 heads)
- **Dataset**: MNIST (60,000 train, 10,000 test)
- **Batch Size**: 32 (effective: 32 × 2 DP = 64)
- **Parallelism**: 2 Data × 2 Tensor × 2 Pipeline

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_data_parallel.py -v
```

## 🛠️ Development

### Adding a New Strategy

1. Create a new strategy in `strategy/strategies/`
2. Inherit from `BaseParallelismStrategy`
3. Implement `apply()` method
4. Register in `strategy/__init__.py`

```python
class MyStrategy(BaseParallelismStrategy):
    def apply(self, model: nn.Module) -> nn.Module:
        # Your parallelism logic here
        return wrapped_model
```

## 📚 Documentation

**📖 [Complete Training Guide](docs/TRAINING_GUIDE.md)** - Detailed walkthrough with diagrams explaining:
- Device Mesh and Process Groups
- Model Wrapping Pipeline (TP → PP → DP)
- Data Flow Architecture
- 1F1B Pipeline Schedule
- Gradient Synchronization

### Key Source Files:

- `parallelism/pipeline_parallel/schedule.py` - 1F1B schedule implementation
- `core/communication.py` - Distributed primitives with autograd support
- `parallelism/data_parallel/core/ddp.py` - DDP implementation details
- `parallelism/tensor_parallel/layers.py` - Column/Row parallel layers


---

<p align="center">
  <strong>Built for learning and scaling deep learning 🧠</strong>
</p>