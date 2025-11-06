# QuintNet

**Distributed Deep Learning with 3D Parallelism**

## Overview

QuintNet is a PyTorch-based library for distributed deep learning. It provides implementations of 3D parallelism: Data Parallelism, Pipeline Parallelism, and Tensor Parallelism. This library is intended for researchers and developers working on large-scale model training. QuintNet is currently in an alpha stage of development.

## Features

*   **Data Parallelism:** Replicates the model across multiple devices, with each device processing a different subset of the data.
*   **Pipeline Parallelism:** Splits the model into stages, with each stage running on a different device.
*   **Tensor Parallelism:** Splits individual layers of the model across multiple devices.
*   **Hybrid 3D Parallelism:** A combination of Data, Pipeline, and Tensor parallelism for training very large models.

## Installation

1.  **Install PyTorch:** It is recommended to install PyTorch with the correct CUDA support using conda. For example:
    ```bash
    conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1
    ```

2.  **Install QuintNet:** You can install QuintNet and its dependencies from this directory using pip:
    ```bash
    pip install -e .
    ```

3.  **Install other dependencies:** The `requirements.txt` file contains a list of additional dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `examples` directory contains scripts that demonstrate how to use the different parallelism features. These examples are currently under development.

To run the examples, use `torchrun`. The number of processes (`nproc_per_node`) should be adjusted based on the type of parallelism and the number of available GPUs.

*   **Data Parallelism:**
    ```bash
    torchrun --nproc_per_node=4 python examples/simple_dp.py
    ```

*   **Pipeline Parallelism:**
    ```bash
    torchrun --nproc_per_node=4 python examples/simple_pp.py
    ```

*   **Tensor Parallelism:**
    ```bash
    torchrun --nproc_per_node=2 python examples/simple_tp.py
    ```

*   **3D Parallelism:**
    ```bash
    torchrun --nproc_per_node=8 python examples/full_3d.py
    ```

## Running Tests

The project uses `pytest` for testing. To run the test suite, navigate to the project's root directory and run the following command:

```bash
pytest
```

## License

QuintNet is licensed under the MIT License.