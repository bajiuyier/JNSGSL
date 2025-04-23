# JNSGSL
Implementation of the JNSGSL Method

## Project Structure Overview

This repository contains the implementation of the **JNSGSL** (Joint Node-Structure Graph Structure Learning) method. Below is an overview of the project directory structure and the function of each major component:

### `config/`
Contains configuration files and utilities for setting hyperparameters and runtime options.

- **`config/jnsgsl/`**: Stores configuration files for each dataset used in experiments.
- **`config/util.py`**: Provides helper functions related to configuration loading and parsing.

### `data/`
Includes all dataset-related resources and utilities.

- **`data/dataset/`**: Stores raw datasets.
- **`data/dataset_utils/`**: Contains preprocessing tools for datasets.
  - `control_homophily.py`: Functions to control the homophily level of a dataset.
  - `get_embedding.py`: Tools for extracting node embeddings.
  - `pyg_load.py`: Dataset loading utilities using PyTorch Geometric.
  - `split.py`: Dataset split functions for training/testing.
  - `dataset.py`: Dataset wrapper that integrates loading and preprocessing steps.

### Top-Level Python Scripts

- **`base_model.py`**: Defines base model classes, including standard GNNs like GCN.
- **`base_solver.py`**: Implements the general workflow for downstream tasks. The `JNSGSLSolver` inherits and extends this base class.
- **`dataset.py`**: High-level interface for dataset loading and transformation.
- **`jnsgsl.py`**: Main implementation of the JNSGSL method, including graph structure learning and downstream task execution.
- **`model.py`**: Contains core class definitions used in the JNSGSL model, including GSL operations, contrastive loss, and embedding updates.
- **`utils.py`**: A set of utility functions used during computation, such as inner product operations and matrix transformation utilities.

