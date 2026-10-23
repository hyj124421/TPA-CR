# TPA-CR

TPA-CR 是一种面向纵向电子病历的药物推荐方法，结合诊断与手术信息、药物共现关系、真实就诊时间间隔和历史处方检索，并使用患者内对比学习辅助训练。

本仓库提供模型、训练和数据预处理代码。受数据使用协议限制，原始 MIMIC 数据、处理后的患者数据和模型权重不随代码发布。

## 项目结构

```text
TPA-CR/
├── README.md
├── src/
│   ├── train.py                 # 训练与验证入口
│   ├── model_net.py             # 模型实现
│   ├── outer_models.py          # 辅助网络与评估指标
│   ├── data_loader.py           # 数据批处理与填充
│   ├── util.py                  # 工具函数
│   └── dataset_jaccard.py       # 可选的数据相似度分析
└── datas/
    ├── mimic-iii/
    │   └── processing_mimic-iii.py
    └── mimic-iv/
        └── processing_mimic-iv.py
```

## 环境

实验环境为 Python 3.8、PyTorch 1.11.0、CUDA 11.3 和 NVIDIA RTX 3080 GPU。代码还依赖 NumPy、pandas、scikit-learn、dill、DGL 和 RDKit；可选的数据相似度分析依赖 matplotlib。

## 数据处理

MIMIC-III 预处理需要以下原始临床表：

- `PRESCRIPTIONS.csv`
- `DIAGNOSES_ICD.csv`
- `PROCEDURES_ICD.csv`
- `ADMISSIONS.csv`

还需要 `atc3toSMILES.pkl`、`ndc2atc_level4.csv`、`drug-atc.csv`、`ndc2rxnorm_mapping.txt` 和 `drug-DDI.csv`。将这些文件放入 `datas/mimic-iii/`，然后从项目根目录运行：

```bash
cd src
python ../datas/mimic-iii/processing_mimic-iii.py
cd ..
```

预处理完成后，将以下文件放入 `datas/mimic-iii/`：

- `voc_final.pkl`
- `records_final.pkl`
- `ehr_adj_final.pkl`
- `ddi_A_final.pkl`

对于 MIMIC-IV，将小写命名的四个临床表及上述五个药物辅助文件放入 `datas/mimic-iv/`，然后运行：

```bash
cd datas/mimic-iv
python processing_mimic-iv.py
cd ../..
```

MIMIC-IV 的主要输出为 `voc_final_4.pkl`、`records_final_4.pkl`、`ehr_adj_final_4.pkl` 和 `ddi_A_final_4.pkl`。使用 MIMIC-IV 训练时，需要在 `src/train.py` 中修改对应的数据加载路径。

## 运行代码

默认使用 MIMIC-III。在项目根目录运行：

```bash
python src/train.py --epochs 5 --lambda_weight 0.02 --tau 0.06 --alpha 1.0 --lambda_schedule fixed
```

查看全部命令行参数：

```bash
python src/train.py --help
```

训练程序优先使用 CUDA，并在 CUDA 不可用时回退到 CPU。模型使用 Adam 优化器，学习率为 `0.0001`，批大小为 1，患者内负样本数为 `K = 3`。验证集 F1 提升时，模型检查点保存至 `state_dict/best_model.pt`。
