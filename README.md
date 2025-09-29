## mAb-Affinity-pred

Analysis, sequence selection and prediction of antibody affinity for monoclonal antibody (mAb) variants from antibody repertoire data. This repository accompanies the preprint:

- Antibody affinity engineering using antibody repertoire data and machine learning — see the preprint on bioRxiv: [https://www.biorxiv.org/content/10.1101/2025.01.10.632313v1](https://www.biorxiv.org/content/10.1101/2025.01.10.632313v1)

### Repository layout
- `VDJ_Sequence_Selection/`: scripts for CDR3/VDJ similarity, clustering, and network visualizations
- `GP_implementation/`: Gaussian Process and baseline model training, nested CV evaluation, plotting utilities
  - `Regression_Evaluation_framework/Regression_evaluation_paramTuning.py`: nested CV + hyperparameter tuning and evaluation
- `Predict_natural_vars/`: training on designed variants and predicting native repertoire variants
- `Analysis_for_publication_R/`: plotting and statistical analysis for figures
- `utils/`: shared helper functions (e.g., `GP_fcts.py`, `NW_functions.py`)

### Requirements
- Python 3.9+
- Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`
- Optional (for network/sequence analyses): `networkx`, `python-Levenshtein`, `tqdm`, `stringdist`


### Data
This code expects CSV/TSV inputs referenced in the scripts (e.g., designed variant KD measurements, VDJ repertoires, MIXCR outputs). 


### Quick start
- Code for CDR3/VDJ sequence selection and clustering utilities under `VDJ_Sequence_Selection/`.

- Evaluate models with nested CV and generate plots:
```bash
python GP_implementation/Regression_Evaluation_framework/Regression_evaluation_paramTuning.py
```

- Train on designed variants and predict native variants (see script for paths/flags):
```bash
python Predict_natural_vars/Train_model_predict_native.py
```

- Preprocess MIXCR/VDJ files (example):
```bash
python Predict_natural_vars/preprocess_VDJ_file.py
```



### Citing
If you use this code, please cite the preprint:

Erlach et al., bioRxiv (2025). Predicting affinity in monoclonal antibody variants and native repertoires. [https://www.biorxiv.org/content/10.1101/2025.01.10.632313v1](https://www.biorxiv.org/content/10.1101/2025.01.10.632313v1)

