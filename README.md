
# 🛰️ Hyperspectral Change Detection with FDIEG-UNet

This repository provides the implementation of our change detection framework on hyperspectral imagery. The model supports patch-based training and is evaluated on datasets such as **Santa Barbara**.

## 📁 Project Structure

```
├── train_cd.py              # Main training script
├── data_loaders.py          
├── dataprocess.py              
├── evaluator.py   
├── trainer.py         
├── losses.py       
├── main.py                
├── models/                  # Network architectures (e.g., networks)
├── datasets/                # Dataset
├── misc/                    # Utilities    
├── checkpoints/             # Saved models
└── vis/                     # Training logs and visualizations
```

## 🛠️ Setup

### 1. Install Dependencies

Make sure you have Python 3.8+ and install required packages:

Or with conda:

```bash
conda create -n cd_env python=3.8
conda activate cd_env
```

### 2. Prepare Dataset

1. [Farmland dataset](https://rslab.ut.ac.ir/data)

2. [Santa Barbara dataset](https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset)

3. [Bay Area dataset](https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset)


Ensure your dataset (e.g., **Santa Barbara**) is placed in the following structure:

```
datasets/
└── santaBarbara/
```

Modify `dataprocess.py` or the `data_loaders.py` if needed.

## 🚀 Training

Run the following command to start training:

```bash
python train_cd.py --patches 9 --project_name Santa_p9_8 --max_epochs 60 --lr_policy step --batch_size 8 --data_name santaBarbara
```

### 🔧 Argument Descriptions

| Argument         | Description                                                  |
|------------------|--------------------------------------------------------------|
| `--patches`      | Number of image patches (default: 9)                         |
| `--project_name` | Project name for logging and saving models                   |
| `--max_epochs`   | Maximum number of training epochs                            |
| `--lr_policy`    | Learning rate policy, e.g., `step`, `cosine`                 |
| `--batch_size`   | Batch size for training                                      |
| `--data_name`    | Dataset name (e.g., `santaBarbara`)                          |

Training results will be saved under:
```
checkpoints/Santa_p9_8/
```

## 📈 Evaluation
Modify the last line of the `train_cd.py`
```bash
python train_cd.py --patches 9 --project_name Santa_p9_8 --max_epochs 60 --lr_policy step --batch_size 8 --data_name santaBarbara
```

## 📋 Citation

If you use this code in your research, please cite our paper (BibTeX will be provided upon publication).

