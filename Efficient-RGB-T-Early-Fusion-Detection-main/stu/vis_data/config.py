auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV'
dataset_type = 'M3FDDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        in_channels=3,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNetRGBTEarlyModifiedStem'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_dfl=dict(loss_weight=0.25, type='DistributionFocalLoss'),
        num_classes=6,
        reg_max=16,
        stacked_convs=4,
        type='GFLHead'),
    corekd_cfg=dict(loss_weight=1, type='MSELoss'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        pad_size_divisor=32,
        rgb_mean=[
            128.2,
            129.3,
            125.3,
        ],
        rgb_std=[
            49.1,
            50.2,
            53.5,
        ],
        thermal_mean=[
            84.1,
            84.1,
            84.1,
        ],
        thermal_std=[
            50.6,
            50.6,
            50.6,
        ],
        type='RGBTDetDataPreprocessor'),
    eval_teacher=True,
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    teacher_ckpt=
    '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/work_dirs/exp1/epoch_12.pth',
    teacher_config=
    '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/configs/gfl/gfl_r50_fpn_1x_m3fd.py',
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=9, type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='KDGFLCLIP')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.01, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
teacher_ckpt = '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/work_dirs/exp1/epoch_12.pth'
teacher_config = '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/mmdetection/configs/gfl/gfl_r50_fpn_1x_m3fd.py'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV/coco/infrared_test.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=
        '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV',
        pipeline=[
            dict(backend_args=None, type='LoadRGBTImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='ResizeRGBT'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetRGBTInputs'),
        ],
        test_mode=True,
        type='M3FDDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV/coco/infrared_test.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadRGBTImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        512,
    ), type='ResizeRGBT'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetRGBTInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        ann_file=
        '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV/coco/infrared_train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=
        '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV',
        filter_cfg=dict(filter_empty_gt=True, min_size=5),
        pipeline=[
            dict(backend_args=None, type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='ResizeRGBT'),
            dict(prob=0.5, type='RandomFlipRGBT'),
            dict(type='PackDetRGBTInputs'),
        ],
        type='M3FDDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadRGBTImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        512,
    ), type='ResizeRGBT'),
    dict(prob=0.5, type='RandomFlipRGBT'),
    dict(type='PackDetRGBTInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV/coco/infrared_test.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=
        '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV',
        pipeline=[
            dict(backend_args=None, type='LoadRGBTImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='ResizeRGBT'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetRGBTInputs'),
        ],
        test_mode=True,
        type='M3FDDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/root/workspace/personal_data/pky/RGBT/Efficient-RGB-T-Early-Fusion-Detection-main/UAV/coco/infrared_test.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/exp2'
