# drone-detection-baseline
This repository serves as the algorithmic baseline for a UAV dataset containing visible and infrared images.

# RSDet ʹ��˵��

## ѵ��
train��python tools/train.py configs/my_config.py --work-dir ./work_dirs/exp1 --amp --auto-scale-lr --resume auto
���������ݼ�����Ҫ��configs/fusion/Resdet�й����µ�ѵ������ ��configs/_base/dataset�й����µ��ļ�

test��python tools/test.py configs/faster_rcnn_r50_fpn_1x_coco.py work_dirs/faster_rcnn/latest.pth
