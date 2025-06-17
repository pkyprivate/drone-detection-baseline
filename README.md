# drone-detection-baseline
This repository serves as the algorithmic baseline for a UAV dataset containing visible and infrared images.

# RSDet 使用说明

## 训练
train：python tools/train.py configs/my_config.py --work-dir ./work_dirs/exp1 --amp --auto-scale-lr --resume auto
对于新数据集，需要在configs/fusion/Resdet中构建新的训练方法 在configs/_base/dataset中构建新的文件

test：python tools/test.py configs/faster_rcnn_r50_fpn_1x_coco.py work_dirs/faster_rcnn/latest.pth
