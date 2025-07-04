backend_args = None
classes = 'UAV'
data_root = '/root/workspace/personal_data/pky/RGBT/UAV'
dataset_type = 'MultispectralDataset'
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
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    FeaFusion=dict(
        channel_nums=[
            256,
            512,
            1024,
            2048,
        ],
        feature_nums=3,
        imgshape=(
            1024,
            1280,
        ),
        loss_MI=dict(input_channels=259, type='MutualInfoLoss'),
        neck=dict(
            in_channels=[
                256,
                512,
                1024,
                2048,
            ],
            num_outs=5,
            out_channels=259,
            type='FPN'),
        num_gate=5,
        num_ins=4,
        scale=[
            4,
            8,
            16,
            32,
        ],
        type='Conv11_Fusion3'),
    Gcommon=dict(
        backbone=dict(
            depth=50,
            frozen_stages=-1,
            init_cfg=dict(
                checkpoint=
                '/root/workspace/personal_data/pky/RGBT/RSDet-master/resnet50-0676ba61.pth',
                type='Pretrained'),
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
            type='ResNet'),
        loss_MI1=dict(input_channels=512, type='MutualInfoLoss'),
        loss_MI2=dict(input_channels=1024, type='MutualInfoLoss'),
        neck=dict(
            in_channels=[
                256,
                512,
                1024,
                2048,
            ],
            num_outs=5,
            out_channels=256,
            type='FPN'),
        strides=[
            4,
            8,
            16,
            32,
            64,
        ],
        type='CommonFeatureGenerator2'),
    Gmask=dict(
        imgshape=(
            1024,
            1280,
        ),
        keep_low=True,
        patch_num=20,
        type='UniqueMaskGenerator3'),
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=
            '/root/workspace/personal_data/pky/RGBT/RSDet-master/resnet50-0676ba61.pth',
            type='Pretrained'),
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
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            51.00417706831775,
            48.91707563987938,
            33.50014617260082,
        ],
        mean_lwir=[
            71.41202617640505,
            71.41202617640505,
            71.41202617640505,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            49.80811692248811,
            49.63865370391049,
            46.54689609388755,
        ],
        std_lwir=[
            47.068441081747956,
            47.068441081747956,
            47.068441081747956,
        ],
        type='PairedDetDataPreprocessor'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=259,
            loss_bbox=dict(beta=1, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            out_channels=259,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                1,
                2,
                3,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=259,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=2000,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.01),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.99, type='nms'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='RSDet_14th')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=25,
        gamma=0.1,
        milestones=[
            6,
            10,
        ],
        type='MultiStepLR'),
]
resume = True
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='Annotation_test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/root/workspace/personal_data/pky/RGBT/UAV',
        metainfo=dict(classes='UAV'),
        pipeline=[
            dict(to_float32=True, type='LoadPairedImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1024,
                    1280,
                ),
                type='PairedImagesResize'),
            dict(size_divisor=32, type='PairedImagesPad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'img_lwir_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackPairedImagesDetInputs'),
        ],
        test_mode=True,
        type='MultispectralDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/root/workspace/personal_data/pky/RGBT/UAV/Annotation_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(to_float32=True, type='LoadPairedImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1280,
    ), type='PairedImagesResize'),
    dict(size_divisor=32, type='PairedImagesPad'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'img_lwir_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackPairedImagesDetInputs'),
]
train_cfg = dict(max_epochs=16, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='Annotation_train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='/root/workspace/personal_data/pky/RGBT/UAV',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes='UAV'),
        pipeline=[
            dict(to_float32=True, type='LoadPairedImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                keep_ratio=True,
                scale=(
                    1024,
                    1280,
                ),
                type='PairedImagesResize'),
            dict(prob=0.5, type='PairedImageRandomFlip'),
            dict(size_divisor=32, type='PairedImagesPad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'img_lwir_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackPairedImagesDetInputs'),
        ],
        type='MultispectralDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(keep_ratio=True, scale=(
        1024,
        1280,
    ), type='PairedImagesResize'),
    dict(prob=0.5, type='PairedImageRandomFlip'),
    dict(size_divisor=32, type='PairedImagesPad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'img_lwir_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackPairedImagesDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='Annotation_test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/root/workspace/personal_data/pky/RGBT/UAV',
        metainfo=dict(classes='UAV'),
        pipeline=[
            dict(to_float32=True, type='LoadPairedImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1024,
                    1280,
                ),
                type='PairedImagesResize'),
            dict(size_divisor=32, type='PairedImagesPad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'img_lwir_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackPairedImagesDetInputs'),
        ],
        test_mode=True,
        type='MultispectralDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/root/workspace/personal_data/pky/RGBT/UAV/Annotation_test.json',
    backend_args=None,
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
work_dir = './work_dirs/UAV'
