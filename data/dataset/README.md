
# JNSGSL Dataset Preparation Instructions

Due to the large size of the datasets, we only provide compressed packages. If you want to download the datasets, please follow the steps below.

1. **Download the dataset**  
   Use the script located at:
   ```
   data/dataset_utils/pyg_load.py
   ```
   to automatically download the required datasets.

2. **Process the dataset**  
   After downloading, process the raw dataset using:
   ```
   data/dataset_utils/dataset.py
   ```

3. **Generate structural feature matrix**  
   To generate structural features (e.g., from random walks via DeepWalk), run:
   ```
   data/dataset_utils/get_embdedding.py
   ```
   The generated structural feature matrix will be saved under:
   ```
   data/dataset/deepwalk_embedding/<dataset_name>
   ```
   Replace `<dataset_name>` with the actual dataset name (e.g., `Cora`, `Citeseer`, etc.).

Please ensure all dependencies are installed before running the scripts. This setup ensures efficient and reproducible preprocessing of the datasets for the JNSGSL framework.

## 🔧 Environment Setup

- **Python version**: 3.9.18  
- **pip version**: 25.0.1  

The following key libraries are required to run this project:

| Package             | Version              | Description                                             |
|---------------------|----------------------|---------------------------------------------------------|
| `torch`             | 2.0.1+cu118          | Core deep learning framework (with CUDA 11.8 support)   |
| `torch-geometric`   | 2.4.0                | Graph neural network library for PyTorch               |
| `torch-scatter`     | 2.1.2+pt20cu118      | Element-wise operations for GNNs                        |
| `torch-sparse`      | 0.6.18+pt20cu118     | Sparse matrix operations for GNNs                       |
| `torch-cluster`     | 1.6.3+pt20cu118      | Clustering methods for PyG                              |
| `torch-spline-conv` | 1.2.2+pt20cu118      | Spline-based convolution operators                      |
| `torchvision`       | 0.15.2+cu118         | Vision models and datasets (with CUDA 11.8 support)     |
| `torchaudio`        | 2.0.2+cu118          | Audio support for PyTorch (with CUDA 11.8 support)      |
| `torchmetrics`      | 1.2.0                | Metric computation utilities for PyTorch models         |

You can install these dependencies using `pip` or create a virtual environment with them. It's recommended to use a `requirements.txt` or `conda` environment for consistency.

```
