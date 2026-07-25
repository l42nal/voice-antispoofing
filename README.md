# Voice Anti-Spoofing using LCNN

Author: Vladislav Smirnov

HSE University, Software Engineering

This repository contains my solution for the **Deep Learning** mini-course project.

The task is to build a **Countermeasure (CM)** system for detecting spoofed speech on the **Logical Access (LA)** partition of the **ASVspoof 2019** dataset.

The project is implemented using the provided **PyTorch Project Template**.

---


## Project Structure

```
.
├── src/                    # Source code (model, trainer, datasets, metrics, etc.)
├── train.py                # Model training
├── inference.py            # Inference on the evaluation set
├── requirements.txt        # Python dependencies
├── README.md
│
├── results/
│   ├── vesmirnov.csv               # Final predictions for submission (Achieved the best result on the repository subset, based on the assignment criteria.)
│   └── checkpoint_eer_results.csv  # EER for every saved checkpoint
│
└── saved/                  # Training checkpoints (ignored by git)
```

---

## Experiment Tracking

Training logs, validation metrics, and experiment details are available

in the following Weights & Biases report:

[Open W&B Report](https://wandb.ai/na4242na-hse-university/pytorch_template/reports/ASVspoof-2019-LA-LCNN-Training-Report--VmlldzoxNzU4MjU5MA?accessToken=mhjlfsabjbwqraxxtppx0abqfvdvdhfov20gwrz1mbe0g9yump2gxac7h0xh59ej)

---

## Dataset

The model was trained on the **ASVspoof 2019 Logical Access (LA)** dataset.

The dataset is **not included** in this repository.

---

## Model

The implemented model follows the **LCNN (Light CNN)** architecture described in the course materials.

Main implementation details:

- STFT front-end
- LCNN architecture
- Cross-Entropy loss
- Dropout before the final BatchNorm layer

---

## Training

Training can be started with

```bash
python train.py
```

Model checkpoints were saved into

```
saved/
```

---

## Inference

To generate predictions for the evaluation set:

```bash
python inference.py
```

or with a specific checkpoint:

```bash
python inference.py inferencer.from_pretrained=path/to/checkpoint.pth
```

---

## Results

During training, every saved checkpoint was evaluated using the official `grading.py` script.

The best checkpoint I achieved with 20 epochs:

| Metric | Value |
|--------|------:|
| **Official EER** | **6.8113%** |

A table with EER values for all checkpoints is available in

```
results/checkpoint_eer_results.csv
```

The final submission file is

```
results/vesmirnov.csv
```

---


## Notes

The repository does **not** include:

- ASVspoof 2019 dataset
- trained model checkpoints (`.pth`)
- cached files and logs

since these files are too large for Git and can be reproduced by running the training pipeline.


---

# Acknowledgements

This project was implemented using the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
provided for the course.

The original template includes many additional features (Hydra, WandB/Comet integration,
project structure, logging, configuration management, etc.). Only the task-specific parts
were modified for this project.
