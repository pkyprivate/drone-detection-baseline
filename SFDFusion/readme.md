https://github.com/lqz2/SFDFusion?tab=readme-ov-file

需要使用wandb

在wandb网站上登录后获取API密钥，或者在cfg.yaml文件中将wandb_mode改为offline

这两个版本必须正确，否则报错

python 3.10

cuda 11.8

```python

def check_mask(root: Path, img_list, config: ConfigDict):
    mask_cache = True
    if (root / 'mask').exists():
        for img_name in img_list:
            if not (root / 'mask' / img_name).exists():
                mask_cache = False
                break
    else:
        mask_cache = False
    if mask_cache:
        logging.info('find mask cache in folder, skip saliency detection')
    else:
        logging.info('find no mask cache in folder, start saliency detection')
        saliency = Saliency()
        saliency.inference(src=root / 'ir', dst=root / 'mask', suffix='jpg')
        
```

这段代码中最后一行suffix要改为数据集对应的图像格式

数据集格式参考RoadScene文件夹
