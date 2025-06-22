# drone-detection-baseline
This repository serves as the algorithmic baseline for a UAV dataset containing visible and infrared images.

# RSDet 
ʹ��˵�� https://github.com/Zhao-Tian-yi/RSDet

## ѵ��
train��python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_common_unique_LLVIP_14th.py --work-dir ./work_dirs/exp1 --amp --auto-scale-lr --resume auto

���������ݼ�����Ҫ��configs/fusion/Resdet�й����µ�ѵ������ ��configs/_base/dataset�й����µ��ļ�
## ����
test��python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_common_unique_LLVIP_14th.py work_dirs/faster_rcnn/latest.pth



# Efficient-RGB-T-Early-Fusion-Detection 
ʹ��˵�� https://github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection

## ѵ��
python tools/train.py  /root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/configs/gfl/gfl_r50_fpn_1x_m3fd.py --work-dir ./work_dirs/exp1 --amp 

python tools/train.py  /root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/configs/gfl/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py --work-dir ./work_dirs/exp2 --amp 

���������ݼ�����Ҫ��configs/�й����µ�ѵ������ ��configs/_base/dataset�й����µ��ļ�
## ����
python tools/test.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/epoch_10.pth --work-dir ./runs/inference



# DAMSDet:
ʹ��˵����https://github.com/gjj45/DAMSDet?tab=readme-ov-file
## ѵ��
python tools/train.py -c configs/damsdet/damsdet_r50vd_llvip.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval

## ����
python tools/eval.py -c configs/damsdet/damsdet_r50vd_llvip.yml --classwise -o weights=output/LLVIP/damsdet_r50vd_llvip/best_model


# ICAFusion:
ʹ��˵����https://github.com/chanchanchan97/ICAFusion?tab=readme-ov-file

## ѵ��
python train.py
