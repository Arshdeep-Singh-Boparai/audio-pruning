# Model Compression of Quaternion Convolutional Neural Networks (QCNNs) 🚀

This repository contains implementations of **model compression techniques** for Quaternion Convolutional Neural Networks (QCNNs), focusing on:

- **Passive Filter Pruning**  
- **Knowledge Distillation (KD)**  

We evaluate these methods on two quaternion models: **QCNN14** and **QResNet38**, quaternion counterparts of the well-known **CNN14** and **ResNet38** architectures from PANNs [1].

> ⚡ This repo is adapted from the [PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn) [1].

---

## 🔹 Pruning Quaternion CNNs

Pruning QCNNs involves two main steps:

1. **Compute layer-wise filter importance** using one of three *passive pruning criteria*:  
   - **L1-norm (`l1`)**  
   - **Geometric Median (`gm`)**  
   - **Operator Norm (`op`)**  

2. **Prune and fine-tune** the model using:  
   - A **user-defined pruning ratio** `p`  
   - The selected pruning method (`l1`, `gm`, or `op`)  

📂 See the [`sorted_index_compute`](./sorted_index_compute) folder for details on filter-importance calculation.  

Once importance scores are obtained, pruned models are fine-tuned using the same training pipeline described in [1].

---

### ▶️ Fine-tuning Pruned QCNN14

```bash
# Fine-tune a pruned QCNN14 model
bash ./scripts/pruned_finetuning.sh
```

Edit `pytorch/QModel.py` → class `QCnn14_pruned(...)` to specify:  
- `checkpoint_path_qcnn14 = './980000_iterations.pth'`  
- `p` = pruning ratio (e.g., 0.25, 0.5)  
- `pruning_method = {'l1', 'gm', 'op'}`  
- `path_to_sorted_index = './out/QCnn14/xxxx'`

---

### ▶️ Fine-tuning Pruned QResNet38

```bash
# Fine-tune a pruned QResNet38 model
bash ./scripts/pruned_finetuning.sh
```

Edit `pytorch/QModel.py` → class `QResNet38_pruned(...)` to specify:  
- `checkpoint_path_qresnet = './920000_iterations.pth'`  
- `p` = pruning ratio  
- `pruning_method = {'l1', 'gm', 'op'}`  
- `path_to_qresnet_sorted_index = './out/QResNet38/xxxx'`  

Also set `--model_type='QResNet38_pruned'` in `pruned_finetuning.sh`.

---

## 🔹 Knowledge Distillation (KD)

Pruned networks can be further compressed using **Knowledge Distillation** (KD), where the pruned model (student) learns from the original model (teacher).  

```bash
# Fine-tune a pruned QCNN via KD
bash ./scripts/pruned_KD.sh
```

---

## 🔹 Checkpoints

👉 Pretrained and pruned checkpoints (QCNN14 & QResNet38) will be made available soon.  

[Download Link Placeholder]()

---

## 🔹 Results

📊 Experimental results on pruning and KD for QCNN14 and QResNet38 will be updated soon.

---

## 🔹 Training QCNN14 & QResNet38

To train models from scratch and obtain pretrained weights:

```bash
WORKSPACE="./workspaces/audioset_tagging"   # Default workspace
echo $WORKSPACE

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train     --workspace=$WORKSPACE     --data_type='full_train'     --window_size=1024     --hop_size=320     --mel_bins=64     --fmin=50     --fmax=14000     --model_type='QCnn14_teacher' \   # or 'QResNet38'
    --loss_type='clip_bce'     --balanced='balanced'     --augmentation='mixup'     --batch_size=32     --learning_rate=1e-3     --resume_iteration=0     --early_stop=1000000     --cuda
```

---

## 📚 References

[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley.  
*PANNs: Large-scale pretrained audio neural networks for audio pattern recognition.*  
**IEEE/ACM Transactions on Audio, Speech, and Language Processing**, 28 (2020): 2880–2894.  

---

## ✨ To-Do
- [ ] Upload pretrained checkpoints  
- [ ] Release experimental results (tables & plots)  
- [ ] Add usage examples for pruning + KD  

---

## 📌 License
Include your preferred license here (e.g., MIT, Apache 2.0).

---
