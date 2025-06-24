https://github.com/hukefy/EI2Det

代码版本要求：python=3.8  ， numpy=1.18.5 

这个numpy版本贼容易报错，建议使用python3.10（否则无法安装timm），并修改requirements.txt文件，不要使用多卡

把这句代码“model = torch.nn.DataParallel(model)”改成“model = model.to('cuda:0')”

数据集格式与M3FD数据集相同


