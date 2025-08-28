# Model compression of Quaternion Convolutional Neural Networks [under prepration]

The rep contains model compression techniques, Passive filter Pruning and Knowledge distillation (KD), to compress Quaternion convolutional neural networks (QCNNs). We compress two quaternion models: QCNN14 and QResNet38. These models are quaternion equivalent of CNN14 and ResNet38 models[1].

This repo is adapted from existing repo designed for [PANNs[1]](https://github.com/qiuqiangkong/audioset_tagging_cnn).



## Pruning quaternion CNNs

Pruning quaternion CNNs involve two steps: 
1. Give pre-trained weights of quaternion network, identify the layer-wise importantce of each filters using three passive filter pruning methods (l_1-norm(l1), geometric median(gm) and operator norm(op)).
2. After computing filter importance of all layers, based on user-defined pruning ratio (p) and pruning method (l1,gm,op), obtain the pruned network and then fine-tune.

```Layer-wise fitler importance calculation
Please see **sorted_index_compute** folder for more information.
```

### Obtaining pruned QCNN network and fine-tuning
      - Once importance of layer-wise filters is obtained, then the pruned quaternion model is fine-tuned. The fine-tuning process involves similar pipeline as explained in [1].
      - This repo (pytorch/QMOdel.py) has all the models (QResNet38_pruned, QCnn14_pruned,...)
      
     ``` #run it to fine-tune the pruned QCNN model,
          bash ./scripts/pruned_finetuning.sh 
          #please specify *checkpoint_path_qcnn14 = './980000_iterations.pth', change pruning ratio (p),specify pruning method= {'l1','gm','op'} path_to_sorted_index* in pytorch/QMOdel--class QCnn14_pruned(...)
      ```
 
 
### Obtaining pruned QResNet38 network and fine-tuning
      
     ``` #run below to fine-tune the pruned QResNet38 model,
          bash ./scripts/pruned_finetuning.sh 
          #please specify --model_type='QResNet38_pruned in pruned_finetuning.sh,  *checkpoint_path_qresnet = './920000_iterations.pth', change pruning ratio (p),specify pruning method= {'l1','gm','op'} path_to_qresent_sorted_index = './out/QReSnet38/xxxx' in pytorch/QMOdel--class QResNet38_pruned(...).
      ```

### Compressing QCNNs via Knowledge distillation (KD)

	- After obtaining the pruned network with either of the pruning method (l1,gm,op), the fine-tuning is performed via KD.
	- bash ./scripts/pruned_KD.sh


## Checkpoint download
    [link to download checkpoints]()
    
    
## Results
   (will be upadated soon)
   
   
## Training QCNN14, QResNet38
To obtain pre-trained weights of  QCNN14 or QResNet38, run following:
   ```
   WORKSPACE="./workspaces/audioset_tagging"   # Default argument.
echo $WORKSPACE

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='QCnn14_teacher' \ #or 'QResNet38' 
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --cuda
   ```


# References

[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.   
   
       
