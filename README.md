# HGCN-DrugRec
HGCN-DrugRec: HyperGraph Convolution Network for Drug Combination Recommendation

## Overview
This repository contains code necessary to run HGCN model. HGCN is an end-to-end model mainly based on hypergraph convolutional network (HGCN). We employ a method of department division to process the [MIMIC-IV](https://mimic.mit.edu/docs/iv/modules/hosp/) dataset and demonstrate
the superior performance of our approach, surpassing most existing models.

## Requirements
- torch==2.0.0
- numpy==1.24.2
- dill==0.3.8
- scikit-learn==1.3.2

## Running the code

### Data Preprocessing
- Due to data privacy considerations, you need to download the diagnoses_icd.csv, procedures_icd.csv, prescriptions.csv from the [MIMIC-IV](https://mimic.mit.edu/docs/iv/modules/hosp/) dataset and put them in ./data/input/.
- The division of departments is based on the curr_service field from the [services](https://mimic.mit.edu/docs/iv/modules/hosp/services/) table.
- The data used in this experiment are from the GU, OMED, and ORTHO departments.
- The data preprocessing code is provided in ./data/process_mimic4.py.
- The code for generating the hypergraph is in ./data/hypergraph_generation.py.

### High-level Clarifications on How to Map ATC Code to SMILES
- The original prescriptions.csv file provides ndc to drugname mapping.
- Use the ndc2RXCUI.txt file for ndc to RXCUI mapping.
- Use the RXCUI2atc4.csv file for RXCUI to atc4 mapping, then change atc4 to atc3.
- Use the drugbank_drugs_info.csv file for drug to SMILES mapping.
- atc3 is a coarse-granular drug classification, one atc3 code contains multiple SMILES strings.

### Model Comparation
Traning codes can be found in ./src/
- **Logistic Regression (LR)** is an instance-based L2-regularized classifier.
- **Ensemble Classifier Chain (ECC).** The Classifier Chain (CC) arranges LR classifiers in a chain structure. We implement a CC ensemble with 10 member.
- **RETAIN** is an RNN-based approach and uses attention and gate mechanism to improve prediction accuracyand interpretability.
- **DMNC** models diagnosis and procedure sequences through multi-view learning and uses Memory Augmented Neural Networks (MANN).
- **GAMENet** is based on memory networks with memory bank enhanced by integrated drug usage, DDI graphs and dynamic memory with patient history.
- **COGNet** is an encoder-decoder based generative network which introduces a novel copy-or-predict mechanism.
- **MICRON** is a recurrent residual networks for predicting medication changes. It allows for sequential updating based only on new patient features.
- **MoleRec** employs substructure-aware molecular representation learning to enhance combinatorial drug recommendation.

### Run the Code
```python
python HGCN.py
```
