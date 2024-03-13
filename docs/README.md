# Cancer Detection on Whole Slide Images (WSIs)

## Dataset Information

### 1. TCGA Dataset
- Main public dataset
- https://www.cancer.gov/tcga

### 2. Annotated Dataset
- Additional dataset with segmentation masks from https://arxiv.org/abs/2008.05332

## Workflow Steps

### 1.1 Segmentation and Patch Generation - TCGA (create_patches_fp.py)
- Creation of patches from WSIs of TCGA Public dataset with additional filtering to remove white patches.
```bash
python create_patches_fp.py --source TCGA_SVS_DIR --save_dir patches_RESULTS_DIR --patch_size 256 --seg --patch --stitch --svs_list_file SLIDE_IDs_SELECTED.csv
```
**Inputs**:
- SVS dataset directory - `TCGA_SVS_DIR`
- Output directory  - `patches_RESULTS_DIR`
- Csv file with selected slide_IDs - `SLIDE_IDs_SELECTED.csv`

**Outputs**: 
- Masks
- Patches
- Stitches
- Output csv with segmentation and patching parameters- `process_list_autogen.csv`

### 1.2 Patching - Annotated Dataset (create_patches_annotated.py)
- Creation of patches from WSIs of anootated dataset based on Segmentation masks.
```bash
python create_patches_annotated.py
```
**Inputs**(defined in code):
- SVS dataset directory - `TCGA_SVS_DIR`
- Output directory  - `patches_RESULTS_Annotated_DIR`
- Annotation mask directory - `complete_region_annotation`
- Annotation mapping csv - `annotated_list.csv`

**Outputs**: 
- Patches
- Output csv with patching parameters - `process_list_autogen_annotated.csv`

### 2. Feature Extraction (extract_features_fp.py)
- Feature extraction based on pretrained ResNet model.
```bash
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir patches_RESULTS_DIR --data_slide_dir TCGA_SVS_DIR --csv_path process_list_autogen.csv --feat_dir features_RESULTS_DIR --batch_size 512 --slide_ext .svs
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir patches_RESULTS_Annotated_DIR --data_slide_dir TCGA_SVS_DIR --csv_path process_list_autogen_annotated.csv --feat_dir features_RESULTS_Annotated_DIR --batch_size 512 --slide_ext .svs
```
**Inputs**(defined in code):
- SVS dataset directory - `TCGA_SVS_DIR`
- Output directory  - `features_RESULTS_DIR`
- Patches directory (previous steps output) - `patches_RESULTS_DIR`
- Csv path (previous step output) - `process_list_autogen.csv`

**Outputs**: 
- .h5 files
- .pt files

### 3. Data Split (create_splits_seq.py)
- Train, validation and Test splits creation for each of the 10 folds.
```bash
python create_splits_seq.py --task task_2_tumor_subtyping --seed 1 --label_frac 1.0 --k 10

python create_splits_seq.py --task task_2_tumor_subtyping --seed 1 --label_frac 1.0 --k 5 --patch_level_dataset
```
**Inputs**(defined in code):
- Output directory  - `splits_DIR`
- Csv file with selected slide_IDs - `SLIDE_IDs_SELECTED_COMBINED_DATASETS.csv`

**Outputs**: 
- Csv files with splits for each fold

### 4. Model Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code task_2_tumor_subtyping_CLAM_100 --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --log_data --data_root_dir features_RESULTS_DIR_Combined_Datasets --results_dir results_DIR --subtyping
```
**Inputs**:
- Features data directory - `features_RESULTS_DIR_Combined_Datasets`
- Output directory  - `results_DIR`
- Csv path (Defined in code) - `SLIDE_IDs_SELECTED_COMBINED_DATASETS.csv`
- Splits directory (Defined in code) - `splits_DIR`

**Outputs**: 
- Results for each split
- Summary csv

### 5. Heat map creation
```bash
CUDA_VISIBLE_DEVICES=0,1 python create_heatmaps.py --config subtyping_config.yaml
```
**Inputs**(defined in code):
- Config yaml file - `subtyping_config.yaml`

**Outputs**: 
- Heatmaps
- TopK patches
