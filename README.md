# Drug–Drug Interaction (DDI) Prediction with Ensemble Learning

This repository contains an implementation of a Drug–Drug Interaction (DDI) prediction model using graph-based similarity measures, label propagation, and an ensemble of base models whose weights are optimized via a genetic algorithm (DEAP).

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
