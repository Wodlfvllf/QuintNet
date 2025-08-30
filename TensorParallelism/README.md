# QuintNet TensorParallelism

A high-performance PyTorch library for tensor parallelism in distributed deep learning. Designed for seamless integration with existing PyTorch workflows while providing efficient scaling across multiple GPUs.

Data Parallelism:     Model Parallelism:      Tensor Parallelism:
┌─────┬─────┬─────┐   ┌─────┐ ┌─────┐        ┌───┬───┬───┬───┐
│  M  │  M  │  M  │   │  L1 │ │  L2 │   →    │ L │ L │ L │ L │
│  o  │  o  │  o  │   │  L2 │ │  L3 │        │ 1 │ 1 │ 1 │ 1 │
│  d  │  d  │  d  │   │  L3 │ │  L4 │        └───┴───┴───┴───┘
│  e  │  e  │  e  │   └─────┘ └─────┘        Split within layer
│  l  │  l  │  l  │   Sequential             Parallel execution
└─────┴─────┴─────┘   execution
Different data        Different layers        Same layer split


## ✨ Key Features

- **🚀 One-Line Integration**: Convert any PyTorch model with a single function call
- **⚡ Optimized Communication**: Custom autograd functions for minimal overhead
- **🔧 Modular Architecture**: Clean, maintainable codebase with clear separation of concerns  
- **📈 Linear Scaling**: Efficient memory and compute distribution across GPUs
- **🛡️ Production Ready**: Robust error handling and comprehensive testing

## 🚀 Quick Start

```python
import torch.distributed as dist
from QuintNet.TensorParallelism import apply_tensor_parallel

# Initialize distributed training
dist.init_process_group(backend="nccl")

# Convert your model to tensor parallel
model = apply_tensor_parallel(model, tp_size=4)

# Continue training as usual - no code changes needed!
```

```bash
# Run distributed training
torchrun --standalone --nproc_per_node=4 your_training_script.py
```

## 🏗️ Architecture Overview

```
QuintNet/TensorParallelism/
├── comm_ops.py          # Distributed communication primitives
├── layers.py            # Tensor parallel layer implementations  
├── rewrite.py           # Automatic model conversion utilities
├── processgroup.py      # Process group management
└── utils.py             # Optional helper functions
```

### Core Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Communication Ops** | Distributed primitives with autograd support | `All_Gather`, `All_Reduce`, `ReduceScatter` |
| **Parallel Layers** | Drop-in replacements for standard layers | `ColumnParallelLinear`, `RowParallelLinear` |
| **Model Rewriter** | Automatic model conversion | `apply_tensor_parallel()` |
| **Process Groups** | Multi-GPU coordination | `ProcessGroupManager` |

## 📊 Performance Benefits

- **Memory Efficiency**: 4x memory reduction with 4-GPU tensor parallelism
- **Communication Optimized**: Minimal overhead with fused operations
- **Scalable**: Linear performance scaling up to model capacity limits

## 🔧 Installation & Setup

```bash
# Prerequisites
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone and setup
git clone <repository-url>
cd QuintNet
```

## 📚 Documentation

- **[Complete Documentation](DOCUMENTATION.md)** - Detailed API reference and advanced usage
- **[Examples](examples/)** - Working examples and tutorials
- **[Performance Guide](docs/performance.md)** - Optimization tips and benchmarks

## 🎯 Use Cases

- **Large Language Models**: Efficient training of transformer architectures
- **Computer Vision**: Scaling ResNets, Vision Transformers, and CNNs
- **Research**: Rapid prototyping of distributed training strategies
- **Production**: Reliable multi-GPU training pipelines

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && isort .
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/your-org/QuintNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/QuintNet/discussions)
- **Documentation**: [Full Documentation](DOCUMENTATION.md)

## 🏆 Acknowledgments

Built with ❤️ for the PyTorch distributed training community.

---

**Ready to scale your models?** Check out our [Complete Documentation](DOCUMENTATION.md) to get started!