
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
```
