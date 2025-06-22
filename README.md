# drone-detection-baseline
This repository serves as the algorithmic baseline for a UAV dataset containing visible and infrared images.

# RSDet 
使用说明 https://github.com/Zhao-Tian-yi/RSDet

## 训练
train：python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_common_unique_LLVIP_14th.py --work-dir ./work_dirs/exp1 --amp --auto-scale-lr --resume auto

对于新数据集，需要在configs/fusion/Resdet中构建新的训练方法 在configs/_base/dataset中构建新的文件
## 测试
test：python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_common_unique_LLVIP_14th.py work_dirs/faster_rcnn/latest.pth



# Efficient-RGB-T-Early-Fusion-Detection 
使用说明 https://github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection

## 训练
python tools/train.py  /root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/configs/gfl/gfl_r50_fpn_1x_m3fd.py --work-dir ./work_dirs/exp1 --amp 

python tools/train.py  /root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/configs/gfl/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py --work-dir ./work_dirs/exp2 --amp 

对于新数据集，需要在configs/中构建新的训练方法 在configs/_base/dataset中构建新的文件
## 测试
python tools/test.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/epoch_10.pth --work-dir ./runs/inference



# DAMSDet:
使用说明：https://github.com/gjj45/DAMSDet?tab=readme-ov-file
## 训练
python tools/train.py -c configs/damsdet/damsdet_r50vd_llvip.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval

## 测试
python tools/eval.py -c configs/damsdet/damsdet_r50vd_llvip.yml --classwise -o weights=output/LLVIP/damsdet_r50vd_llvip/best_model


# ICAFusion:
使用说明：https://github.com/chanchanchan97/ICAFusion?tab=readme-ov-file

## 训练
python train.py
