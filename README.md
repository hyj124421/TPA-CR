# TPA-CR

TPA-CR is a medication recommendation method for longitudinal electronic health records. It integrates diagnosis and procedure information, medication co-occurrence relationships, actual time intervals between visits, and historical prescription retrieval, with intra-patient contrastive learning as an auxiliary training objective.

This repository provides the model implementation, training scripts, and data preprocessing code. Due to data use agreements, raw MIMIC data, processed patient data, and model weights are not included.
## Project Structure

```text
TPA-CR/
├── README.md
├── src/
│   ├── train.py                 # Training and validation entry point
│   ├── model_net.py             # Model implementation
│   ├── outer_models.py          # Auxiliary networks and evaluation metrics
│   ├── data_loader.py           # Data batching and padding
│   ├── util.py                  # Utility functions
│   └── dataset_jaccard.py       # Optional dataset similarity analysis
└── datas/
    ├── mimic-iii/
    │   └── processing_mimic-iii.py
    └── mimic-iv/
        └── processing_mimic-iv.py
```

## Environment

Experiments were conducted using Python 3.8, PyTorch 1.11.0, CUDA 11.3, and an NVIDIA RTX 3080 GPU. Additional dependencies include NumPy, pandas, scikit-learn, dill, DGL, and RDKit. The optional dataset similarity analysis requires matplotlib.

## Data Preprocessing

MIMIC-III preprocessing requires the following raw clinical tables:

- `PRESCRIPTIONS.csv`
- `DIAGNOSES_ICD.csv`
- `PROCEDURES_ICD.csv`
- `ADMISSIONS.csv`

The following auxiliary files are also required: atc3toSMILES.pkl, ndc2atc_level4.csv, drug-atc.csv, ndc2rxnorm_mapping.txt, and drug-DDI.csv.
Place these files in datas/mimic-iii/, then execute the following commands from the project root:

```bash
cd src
python ../datas/mimic-iii/processing_mimic-iii.py
cd ..
```

After preprocessing, place the following generated files in datas/mimic-iii/:

- `voc_final.pkl`
- `records_final.pkl`
- `ehr_adj_final.pkl`
- `ddi_A_final.pkl`

For MIMIC-IV, place the four clinical tables with lowercase filenames and the five auxiliary medication files listed above in datas/mimic-iv/, then run:

```bash
cd datas/mimic-iv
python processing_mimic-iv.py
cd ../..
```

The main MIMIC-IV outputs are voc_final_4.pkl, records_final_4.pkl, ehr_adj_final_4.pkl, and ddi_A_final_4.pkl. To train on MIMIC-IV, update the corresponding data loading paths in src/train.py.

## Running the Code

MIMIC-III is used by default. Run the following command from the project root:

```bash
python src/train.py --epochs 5 --lambda_weight 0.02 --tau 0.06 --alpha 1.0 --lambda_schedule fixed
```

To view all command-line options:

```bash
python src/train.py --help
```

The training script uses CUDA when available and otherwise falls back to CPU. The model is trained using the Adam optimizer with a learning rate of 0.0001, a batch size of 1, and K = 3 intra-patient negative samples. Whenever the validation F1 score improves, the model checkpoint is saved to state_dict/best_model.pt.
