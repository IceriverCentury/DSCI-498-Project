# DSCI-498-Project

# Project Title: Abusive Language Detection From Joined Modelling

# Project Abstract
## This project implements and extends the paper:
"Joint Modelling of Emotion and Abusive Language Detection" (ACL 2020) by Rajamanickam et al.([Paper Link]([url](https://aclanthology.org/2020.acl-main.394.pdf))).
We build multitask models that jointly learn to detect:
### Offensive/abusive language
### Emotion categories
### Figurative language (e.g., irony)
The goal is to improve abusive language detection by leveraging emotional and figurative cues through multitask learning.

## This study will expand on previous work in two key ways. First, it will incorporate at least one additional recent dataset to ensure a broader evaluation across different contexts. Second, it will introduce an additional auxiliary task beyond emotion detection to further refine the model’s ability to distinguish abusive language. By integrating multiple related tasks, this approach aims to provide deeper insights into the interplay between emotional expression and abusive content, ultimately leading to more robust and context-aware abuse detection models.

Task | Dataset | Source
Abuse Detection | OffensEval-2019 (OLID) | Offensive tweet classification
Emotion Detection | SemEval-2018 Task 1 | Emotion intensity classification
Irony Detection | SemEval-2018 Task 3 | Irony detection in tweets
Optional Additional Abuse Dataset | HateXplain | Multilabel hate speech and offense

# Datasets 
Task | Dataset | Source
Abuse Detection | OffensEval-2019 (OLID) | Offensive tweet classification
Emotion Detection | SemEval-2018 Task 1 | Emotion intensity classification
Irony Detection | SemEval-2018 Task 3 | Irony detection in tweets
Optional Additional Abuse Dataset | HateXplain | Multilabel hate speech and offense
We preprocessed all datasets into two columns: "text" and "label", tokenized them, padded embeddings, and aligned them for model input.


# Required Packages
You can install all required packages via pip:
pip install torch torchvision torchaudio
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install datasets
pip install tqdm

# ⚙️ How to Run
1. Clone the Repo
git clone https://github.com/IceriverCentury/DSCI-498-Project.git
cd your-repo-name
2. Run the script main.ipynb step by step.

# Models Implemented
## STL (Single Task Learning) for abuse detection
## MTL-Gated Double Encoder (primary contribution of the paper)
## Extended MTL with Irony detection as an additional auxiliary task
## Custom attention layers after LSTM encodings
## Sigmoid activations for binary/multilabel outputs
## BCEWithLogitsLoss for stable training

# Future Extensions
## Add contrastive pretraining between abusive and figurative representations.
## Explore dynamic task weighting (uncertainty-based MTL).
## Pretrain the shared encoder using masked language modeling.


