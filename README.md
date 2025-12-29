# Drug–Drug Interaction (DDI) Prediction with Ensemble Learning

This repository contains an implementation of a Drug–Drug Interaction (DDI) prediction model using graph-based similarity measures, label propagation, and an ensemble of base models whose weights are optimized via a genetic algorithm (DEAP).

This project mainly re-implements and refactors the method described in:
“Predicting potential drug-drug interactions by integrating chemical, biological, phenotypic and network data”
(BMC Bioinformatics, 2017). Original article: [https://link.springer.com/article/10.1186/s12859-016-1415-9](https://doi.org/10.1186/s12859-016-1415-9)

---

## Project structure

```text
.
├─ src/
│  └─ ddi_prediction.py                     # Main module      
├─ dataset/    
│  ├─ Mô tả về các đặc trưng/               # Input data
│  ├─ drug_drug_matrix.csv
│  ├─ chem_Jacarrd_sim.csv
│  ├─ target_Jacarrd_sim.csv
│  ├─ transporter_Jacarrd_sim.csv
│  ├─ enzyme_Jacarrd_sim.csv
│  ├─ pathway_Jacarrd_sim.csv
│  ├─ indication_Jacarrd_sim.csv
│  ├─ sideeffect_Jacarrd_sim.csv
│  └─ offsideeffect_Jacarrd_sim.csv
├─ result/                                  # Output
├─ README.md
└─ requirements.txt
```

## Installation

```text
1. Clone this repository:
git clone https://github.com/ankhtn/ddi-prediction.git
2. Install dependencies:
pip install -r requirements.txt
```

## Data description

```text
All matrices in dataset/ are stored as CSV files with:
- The first row as a header (ignored by the loader),
- The first column as an ID / name (also ignored),
- The remaining entries forming a square matrix.
Main files:
- drug_drug_matrix.csv (Binary drug–drug interaction adjacency matrix (1 = interaction, 0 = no known interaction)).
- chem_Jacarrd_sim.csv, target_Jacarrd_sim.csv, transporter_Jacarrd_sim.csv, enzyme_Jacarrd_sim.csv, pathway_Jacarrd_sim.csv, indication_Jacarrd_sim.csv, sideeffect_Jacarrd_sim.csv, offsideeffect_Jacarrd_sim.csv (Pairwise similarity matrices between drugs based on different biological / functional views
(chemical structure, targets, enzymes, pathways, indications, side effects, etc.)).
```

## Usage

```text
From the repository root: python src/ddi_prediction.py
By default, this will:
- Load dataset/drug_drug_matrix.csv,
- Perform 3-fold cross-validation,
- Repeat the experiment for multiple random seeds,
- Write metrics and ensemble weights to the result/ directory, for example:
  + result/result_on_our_dataset_3CV_0.txt
  + result/weights_on_our_dataset_3CV_0.txt
Each result_on_our_dataset_3CV_<seed>.txt contains:
- AUC and AUPR of each base model,
- AUC, AUPR, precision, recall, accuracy and F1 of the ensemble models.
Each weights_on_our_dataset_3CV_<seed>.txt stores the optimized ensemble weights learned by the genetic algorithm.
If you want to change the number of runs, folds or output file prefix, you can modify the parameters of the main() function inside src/ddi_prediction.py.
```

## Methods

```text
The implementation combines:
- Neighborhood-based similarity methods using multiple drug similarity matrices
- Label propagation on similarity graphs
- Topology-based scores on the DDI network:
  + common neighbors
  + Adamic–Adar
  + resource allocation
  + Katz index
  + average commute time (ACT)
  + random walk with restart (RWR)
- A matrix perturbation–based method
- A weighted ensemble of base models where:
  + weights are optimized by a genetic algorithm (DEAP)
  + the optimization objective is to maximize F1 / AUPR on validation data.
Evaluation metrics:
- AUC (ROC)
- AUPR
- Precision, Recall, Accuracy, F1-score
```

## Reference & Acknowledgements

```text
This work is mainly based on and inspired by:

Wen Zhang et al., “Predicting potential drug-drug interactions by integrating chemical, biological, phenotypic and network data”, BMC Bioinformatics, 2017.
Original article: [https://link.springer.com/article/10.1186/s12859-016-1415-9](https://doi.org/10.1186/s12859-016-1415-9)

The goal of this repository is to reproduce and refactor the original method for learning and research purposes.
```
