# QuintNet — Towards 5D Parallelism for Scalable Deep Learning

QuintNet is a research-oriented PyTorch framework designed to explore and implement multi-dimensional parallelism strategies for distributed deep learning. Our goal is to enable efficient training and inference of massive foundation models that exceed the capacity of single GPUs or even single nodes. By combining five distinct parallelism dimensions—**5D Parallelism**—QuintNet provides a modular, extensible toolkit for scaling deep learning workflows across clusters.

Starting with **Data Parallelism** and **Tensor Parallelism** as foundational modules, we plan to progressively integrate Sequence/Context, Pipeline, and Expert Parallelism. This allows users to experiment with hybrid approaches tailored to their models and hardware constraints.

![QuintNet Logo](https://github.com/Wodlfvllf/QuintNet/blob/main/imgs/Quintnet.jpeg)

## 🔥 The 5D Parallelism Strategy

Modern AI models demand unprecedented computational resources. Traditional single-strategy parallelism approaches hit fundamental bottlenecks when scaling to trillion-parameter models. **5D Parallelism** addresses this by distributing computation across five distinct dimensions:

### **1. Data Parallelism (DP) — Along the Batch Dimension**
- **Strategy**: Replicate the model across devices, each processing different data batches
- **Benefits**: High throughput, simple implementation, excellent scaling for large batch sizes
- **Trade-offs**: Memory-intensive (full model replication), communication overhead for gradient synchronization
- **Best For**: Models that fit in single GPU memory, throughput-critical applications

### **2. Tensor Parallelism (TP) — Along the Hidden Dimension**
- **Strategy**: Shard individual layers (weights, activations) across devices within the same forward/backward pass
- **Benefits**: Enables training models larger than single GPU memory, fine-grained parallelism
- **Trade-offs**: High communication frequency, requires careful synchronization
- **Best For**: Large models with memory constraints, low-latency inference

### **3. Sequence/Context Parallelism (SP/CP) — Along the Sequence Dimension**
- **Strategy**: Distribute long sequences across devices, particularly for attention mechanisms
- **Benefits**: Enables extremely long context lengths, reduces memory per device for sequence processing
- **Trade-offs**: Complex attention computation distribution, sequence-dependent operations
- **Best For**: Long-context transformers, document-level processing, time-series models

### **4. Pipeline Parallelism (PP) — Along the Model Layers**
- **Strategy**: Partition model layers across devices, creating a pipeline of forward/backward passes
- **Benefits**: Scales to very deep models, reduces memory per device significantly
- **Trade-offs**: Pipeline bubbles reduce efficiency, requires micro-batching for optimal utilization
- **Best For**: Very deep networks, memory-constrained environments with many devices

### **5. Expert Parallelism (EP) — Along the Model Experts**
- **Strategy**: Route different inputs to specialized expert sub-models distributed across devices
- **Benefits**: Massive parameter scaling with sparse activation, conditional computation
- **Trade-offs**: Load balancing challenges, routing overhead, expert utilization optimization
- **Best For**: Mixture-of-Experts models, extremely large capacity requirements

## 🎯 Why 5D Parallelism Matters

Each parallelism dimension addresses different scaling bottlenecks:

- **Memory Bottlenecks**: TP, PP, and SP reduce per-device memory requirements
- **Communication Bottlenecks**: Strategic combination minimizes cross-device traffic
- **Utilization Bottlenecks**: EP and PP enable better hardware utilization patterns
- **Scalability Bottlenecks**: Combined strategies enable linear scaling to thousands of devices

**QuintNet's Vision**: Enable seamless combination of all five dimensions, automatically optimizing the parallelism strategy based on model architecture, hardware topology, and performance requirements.

## ✨ Key Features & Implementation Status

### **Framework Characteristics**
- **Modular Architecture**: Each parallelism strategy is independently implementable and composable
- **PyTorch-Native**: Built on `torch.distributed` with minimal external dependencies
- **Research-Oriented**: Clean, well-documented implementations for educational and experimental use
- **Production-Ready**: Performance-optimized with real-world deployment considerations

### **Implementation Status**

#### **✅ Completed Modules**
- **Data Parallelism**: Advanced custom DDP implementation with gradient bucketing, modular backends, and optimized communication patterns
- **Tensor Parallelism**: Efficient weight sharding for linear layers with autograd-aware communication primitives (All_Gather, All_Reduce, ReduceScatter)

#### **🚧 In Development**
- **Pipeline Parallelism**: Layer partitioning with micro-batching and bubble minimization strategies

#### **📋 Planned Modules**
- **Sequence/Context Parallelism**: Attention mechanism distribution for long-context models
- **Expert Parallelism**: MoE routing with load balancing and expert placement optimization
- **5D Orchestrator**: Automatic configuration system for optimal parallelism strategy selection

## 📂 Repository Structure

```
QuintNet/
├── QuintNet/                   # Main framework package
│   ├── DataParallelism/        # ✅ Data parallelism implementation
│   │   ├── core/               # Core DDP logic and configuration
│   │   ├── components/         # Modular components (buckets, reducers, broadcasters)
│   │   ├── backends/           # Communication backend abstractions
│   │   ├── utils/              # Factory functions and utilities
│   │   ├── tests/              # Comprehensive test suite
│   │   └── ddp_wrapper.py      # High-level API wrapper
│   ├── TensorParallelism/      # ✅ Tensor parallelism implementation
│   │   ├── comm_ops.py         # Communication primitives
│   │   ├── layers.py           # Parallel layer implementations
│   │   ├── rewrite.py          # Model transformation utilities
│   │   ├── processgroup.py     # Process group management
│   │   └── utils.py            # Helper functions
│   ├── PipelineParallelism/    # 🚧 Pipeline parallelism (in development)
│   ├── SequenceParallelism/    # 📋 Sequence/context parallelism (planned)
│   ├── ExpertParallelism/      # 📋 Expert/MoE parallelism (planned)
│   ├── Orchestrator/           # 📋 5D strategy optimization (planned)
│   ├── imgs/                   # Documentation assets
│   └── requirements.txt        # Framework dependencies
├── Mnist-Digit-Classification/ # Training examples and benchmarks
│   ├── Base_training/          # Single-device baseline
│   ├── Training_Data_Parallelism/     # Data parallel examples
│   ├── Training_Tensor_Parallelism/   # Tensor parallel examples
│   ├── Training_Pipeline_Parallelism/ # Pipeline parallel examples (coming soon)
│   └── utilities/              # Shared training utilities
├── dataset/                    # Example datasets
├── benchmarks/                 # Performance benchmarking suite
└── docs/                       # Comprehensive documentation
```

## 🚀 Quick Start

### Environment Setup
Create and activate a conda environment with PyTorch and distributed training support:

```bash
conda create -n quintnet python=3.11 -y
conda activate quintnet

# Install PyTorch with CUDA support (adjust for your CUDA version)
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

# Install framework dependencies
pip install -r QuintNet/requirements.txt
```

### Training Examples

#### Single Device Baseline
```bash
python -m Mnist-Digit-Classification.Base_training.train
```

#### Data Parallel Training
```bash
# Single node, multiple GPUs
torchrun --standalone --nproc_per_node=4 \
  -m Mnist-Digit-Classification.Training_Data_Parallelism.train_ddp

# Multi-node distributed training
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
  --master_addr="10.0.0.1" --master_port=29500 \
  -m Mnist-Digit-Classification.Training_Data_Parallelism.train_ddp
```

#### Tensor Parallel Training
```bash
# Intra-node tensor parallelism
torchrun --standalone --nproc_per_node=2 \
  -m Mnist-Digit-Classification.Training_Tensor_Parallelism.TP_training
```

## 🏗️ Development Roadmap

### **Phase 1: Foundation (Completed)**
- ✅ Data Parallelism with advanced bucketing and communication optimization
- ✅ Tensor Parallelism with efficient weight sharding and synchronization
- ✅ Comprehensive testing framework and documentation

### **Phase 2: Other Parallelism (In Progress)**
- 🚧 Pipeline Parallelism with micro-batching and schedule optimization
- 📋 Sequence/Context Parallelism for long-context transformer models
- 📋 Expert Parallelism with MoE routing and load balancing

### **Phase 3: Integration and Optimization (Planned)**
- 📋 Hybrid parallelism combinations (DP+TP, TP+PP, etc.)
- 📋 Automatic strategy selection based on model and hardware characteristics
- 📋 Advanced memory optimization techniques (ZeRO-style optimizations)
- 📋 Communication optimization for various network topologies

## 📊 Performance Philosophy

QuintNet prioritizes:

1. **Scalability**: Linear scaling across devices and nodes
2. **Efficiency**: Minimal communication overhead and optimal memory utilization
3. **Flexibility**: Easy combination of parallelism strategies
4. **Debuggability**: Clear error messages and comprehensive logging
5. **Educational Value**: Transparent implementations for learning and research

## 🎓 Educational Resources

### Learning Path
1. **Start with Data Parallelism**: Understand gradient synchronization and distributed optimization
2. **Progress to Tensor Parallelism**: Learn intra-layer parallelism and communication patterns
3. **Explore Pipeline Parallelism**: Master inter-layer parallelism and micro-batching
4. **Advanced Topics**: Sequence parallelism for attention mechanisms and expert parallelism for MoE

### Key Concepts Covered
- Distributed training fundamentals
- Communication primitive implementations
- Gradient synchronization mechanisms
- Memory optimization strategies
- Autograd integration for custom operations
- Process group management and topology awareness

## 🧪 Testing and Validation

QuintNet includes comprehensive testing for:
- **Correctness**: Forward/backward pass numerical consistency
- **Performance**: Communication overhead and scaling efficiency
- **Robustness**: Error handling and fault tolerance
- **Integration**: Compatibility across different parallelism combinations

## 🤝 Contributing

We welcome contributions across all areas:
- **Core Implementations**: New parallelism strategies and optimizations
- **Performance Improvements**: Communication efficiency and memory optimization
- **Documentation**: Tutorials, examples, and API documentation  
- **Testing**: Unit tests, integration tests, and benchmarking
- **Research**: Novel parallelism strategies and hybrid approaches

## 🔬 Research Applications

QuintNet enables research in:
- **Parallelism Strategy Optimization**: Automatic selection and tuning
- **Communication Pattern Analysis**: Understanding bottlenecks in distributed training
- **Memory Efficiency**: Novel approaches to memory optimization in distributed settings

## 🏆 Acknowledgments

QuintNet builds upon foundational work from:
- **PyTorch Distributed**: Core distributed training infrastructure
- **DeepSpeed** (Microsoft): Advanced optimization techniques and ZeRO memory optimization
- **Megatron-LM** (NVIDIA): Pioneering tensor and pipeline parallelism implementations
- **FairScale** (Meta): Modular parallelism components and experimental frameworks
- **Alpa** (UC Berkeley): Automatic parallelization strategies
- **PaLM** (Google): Large-scale training insights and expert parallelism

---

⚡ **From Single GPU to Exascale: Democratizing 5D Parallelism for the AI Community**

*QuintNet empowers researchers, practitioners, and students to explore the cutting edge of distributed deep learning through clean, modular, and high-performance implementations of state-of-the-art parallelism strategies.*