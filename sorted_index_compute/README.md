# Sorted-Index Generator (QCNN14 & QResNet38)

Each script takes  **three arguments**:
1) checkpoint path (`.pth`), 2) output directory, 3) method (`gm`, `op`, or `l1`).

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy torch tqdm
```

## Usage
```bash
# QCNN14 (GM method)
python run_qcnn14.py /path/to/qcnn14_checkpoint.pth outputs/qcnn14 gm

# QResNet38 (OP method)
python run_qresnet38.py /path/to/qresnet38_checkpoint.pth outputs/qresnet38 op
```

## Output
For each target layer, the scripts save:
- `<layer>_mean_score.npy`
- `<layer>_sorted_index.npy` (ascending order) [[output dir: ./sorted_index]]
- a `meta.json` in the output folder
