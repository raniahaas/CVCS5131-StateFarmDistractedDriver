# Learning the Action, Not the Driver

CVCS5131 project on the [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) dataset.
We test whether a ResNet-18 classifies distracted driving better on **unseen drivers** when (1) a driver adversary removes identity from its features (gradient reversal or a confusion loss) and (2) upper-body pose keypoints are fused with the image features. Every configuration is trained from scratch and from ImageNet weights.

## Compute budget

| Notebook | GPU-hours | Hardware |
|---|---:|---|
| `distracted_driver.ipynb` (main study) | **5.4** (5.0 training + 0.4 one-time pose extraction) | NVIDIA RTX 5070 Ti Laptop GPU, 12 GB |
| `adversarial_study.ipynb` (companion study, optional) | **11.9** | same GPU |
| **Total** | **17.3** | |

All runs used torch 2.11, CUDA 12.8 and mixed precision. The classical baselines add 0.03 CPU-hours. The compute budget section of each notebook (Section 13 in the main study, Section 19 in the companion) breaks the time down by stage. A slower GPU takes proportionally longer.

## Files

| Path | Purpose |
|---|---|
| `distracted_driver.ipynb` | Main study. Produces every result in the report |
| `adversarial_study.ipynb` | Companion study that tunes the driver adversary. It chose the confusion-loss settings used in the main study and holds the negative results |
| `outputs/main/`, `outputs/adversarial/` | Tables and figures written by the two notebooks |
| `outputs/archive_with_flip/` | Earlier main-study run with the horizontal flip, quoted in Section 3 of the main notebook |
| `figures/` | Scripts for the paper and presentation diagrams |
| `printout/` | LaTeX paper |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Environment specification |

Both notebooks are committed with their outputs from the final run, so all results can be read without rerunning anything.

## How to run

**1. Install the environment.** Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip: `pip install -r requirements.txt`. Both install the CUDA 12.8 build of PyTorch.

**2. Get the data.** The dataset is not in this repo because the Kaggle terms do not allow redistributing it. Accept the [competition rules](https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules), download the data from the [data page](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data), and extract it so this layout exists:

```
data/state-farm-distracted-driver-detection/
├── driver_imgs_list.csv
└── imgs/train/c0 ... c9/*.jpg
```

Only `driver_imgs_list.csv` and `imgs/train/` are used.

**3. Run the notebook.** Open `distracted_driver.ipynb` in Jupyter or VS Code, select the project's `.venv` kernel, and run all cells. Nothing needs to be changed.

- Tables are written to `outputs/main/` and figures to `outputs/main/figures/`.
- The first run caches the resized images and pose keypoints in `cache/`. The pretrained weights are downloaded by `torchvision`, so internet access is needed.
- An interrupted run resumes where it stopped. Set `RESUME = False` in Section 6.3 to retrain everything.

**Optional.** `adversarial_study.ipynb` reads the caches and fold tables from the main study, so run it after `distracted_driver.ipynb`. Its results go to `outputs/adversarial/`.

## Reproducibility

- **Seeds.** All randomness comes from `SEED = 42` in Section 0 (Python, NumPy and PyTorch). The four frozen test drivers and the driver-grouped folds are derived from it.
- **Evaluation protocol.** The test drivers are never used for training, validation or tuning. The other 22 drivers form driver-grouped cross-validation folds. Each final model is scored once on the test drivers (Section 8.5).
- **Environment.** `uv.lock` pins every package version.

## Reused code

The project started from two Kaggle course notebooks by Emmanuel Pintelas on this competition, "Image-based ML Master - vision classification" (Lab 6, <https://www.kaggle.com/code/emmanuelpintelas/image-based-ml-master-vision-classification>) and "Lec 9 - ML Lab 7 - Transfer Learning with Residual Networks" (Lab 7). They were the starting point for data loading and the ResNet training loop. The team then rewrote and extended the code almost completely, adding the driver-grouped evaluation, the leakage study, pose fusion, the driver adversaries, the identity probes, the scaling analysis, the classical baselines and the error analysis. Model architectures and pretrained weights come from `torchvision`.

## AI usage

Generative AI was used as a coding assistant for the notebooks and scripts in this repository. The authors reviewed the AI-assisted code and take responsibility for its correctness and for the reported results.
