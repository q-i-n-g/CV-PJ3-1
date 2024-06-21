# CV-PJ3-1
在STL-10上使用SimCLR进行自监督学习，随后在CIFAR-100上重新训练线性输出层进行测试。以resnet-18为例，对比了自监督学习与监督学习。

## 数据准备
下载CIFAR-100, STL-10数据集，数据集下载链接如下：

CIFAR-100:

https://www.cs.toronto.edu/~kriz/cifar.html

STL-10:

https://cs.stanford.edu/~acoates/stl10/

如果只需要测试模型，可以只下载CIFAR-100。

下载数据后，在根目录下创建datasets文件夹，将下载好的数据集解压到datasets文件夹中，文件目录格式如下:
'''
root
├── datasets
│   ├── cifar-10-batches-py
│   ├── cifar-100-python
│   │   ├── file.txt~
│   │   ├── meta
│   │   ├── test
│   │   └── train
│   └── stl10_binary
│       ├── class_names.txt
│       ├── fold_indices.txt
│       ├── test_X.bin
│       ├── test_y.bin
│       ├── train_X.bin
│       ├── train_y.bin
│       └── unlabeled_X.bin
└── cifar-100-python.tar.gz
└── stl10_binary.tar.gz
'''

## 模型准备
到Google Drive下载模型权重，直接放置在根目录下，相对位置如下：
'''
root
└── stl10
    ├── tensorboard_logs
    ├── model.pth.tar
    └── plot.png
'''

## 测试
运行
'''
python test.py
'''
将会输出在CIFAR-100测试集上的top1和top5准确率，如果需要测试其他模型，需要手动修改test.py中的model_path参数。

## 训练
### 预训练
运行
'''
python run.py -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 100 --gpu-index 0
'''
将会在STL-10上预训练，会创建一个新的./checkpoint文件夹，并会将预训练好的模型保存在其中。

### 在CIFAR-100上调整线性输出层
运行
'''
python finetune.py
'''
注意，需要修改其中的model_path参数到上述预训练的模型权重model.pth文件的路径下，将会创建一个./runs文件夹，并将调整后的模型保存在其中。也可以自己定义模型保存路径。
