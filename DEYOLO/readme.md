https://github.com/chips96/DEYOLO

# 训练代码
```python
from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/models/v8/DEYOLO.yaml").load("yolov8n.pt")

# Train the model
train_results = model.train(
    data="M3FD.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
```

# 数据集路径组织
'''
Your dataset
├── ...
├── images
|   ├── vis_train
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
|   ├── vis_val
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
|   ├── Ir_train
|   |   ├── 100.jpg
|   |   ├── 101.jpg
|   |   └── ...
|   ├── Ir_val 
|   |   ├── 100.jpg
|   |   ├── 101.jpg
|   |   └── ...
└── labels
    ├── vis_train
    |   ├── 1.txt
    |   ├── 2.txt
    |   └── ...
    └── vis_val
        ├── 100.txt
        ├── 101.txt
        └── ...
'''
