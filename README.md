# Style Transfer
Style Transfer with Tensorflow

## Environments
* OS  : Windows 10 Edu x86-64
* CPU : i7-7700K
* GPU : GTX 1060 6GB
* RAM : DDR4 16GB
* Library : TF 1.5 with CUDA 9.0 + cuDNN 7.0
* Python 3.6

## Prerequisites
* python 3.x
* tensorflow 1.x
* scipy
* pillow
* Internet :) (for downloading VGG19 pre-trained model)

## Usage
    $ python style_transfer.py

## Repo Tree
```
│
├── checkpoints       (model checkpoint)
│    ├── checkpoint
│    ├── ...
│    └── xxx.ckpt
├── contents          (content images)
│    ├── deadpool.jpg
│    └── ...
├── styles            (style images)
│    ├── guernica.jpg
│    └── ...
├── outputs           (results of style-transfer)
│    ├── deadpool_guernica_xxx.jpg
│    └── ...
├── graphs            (tensorboard)
│    └── ...
├── utils.py          (image utils & download pre-trained model)
├── vgg19.py          (VGG19 model)
└── style_transfer.py (style transfer)
```

## Author
HyeongChan Kim / ([@kozistr](https://kozistr.github.io), [@zer0day](http://zer0day.tistory.com))
