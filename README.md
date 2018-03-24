# Style Transfer (Neural Style)
Tensorflow implementation of Image Style Transfer (Neural Style)

## Preferred Environments
* OS  : Windows 10 or Ubuntu 14.04 ~
* GPU : ~
* Library : TF 1.x with CUDA ~ + CuDNN ~
* Python 3.x

## Prerequisites
* Python 3.x
* Tensorflow 1.x
* Scipy
* Pillow
* Internet :) (for downloading VGG19 pre-trained model)

## Usage
    $ python style_transfer.py --content <content image> --style <style image> ...

*Example* : ```python style_transfer.py --content content/deadpool.jpg --style style/guernica.jpg```

### Arguments

*Required*
* ```--content``` : file path of a content image, *default* : ```contents/deadpool.jpg```
* ```--style``` : file path of a style image, *default* : ```styles/guernica.jpg```

*Optional*
* ```--content_w``` : weight of content loss, *default* : ```0.05```
* ```--style_w``` : weight of style loss, *default* : ```0.02```
* ```--image_width``` : file path of a content image, *default* : ```333```
* ```--image_height``` : file path of a style image, *default* : ```250```
* ```--train_steps``` : total training epochs, *default* : ```500```

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

## Sample results

### DeadPool with guernica & monet & vangogh (333 x 250)

*conetent\style* | *guernica* | *monet* | *vangogh*
:---: | :---: | :---: | :---:
![Generated Image](https://github.com/kozistr/style-transfer/blob/master/contents/deadpool.png) | ![Generated Image](https://github.com/kozistr/style-transfer/blob/master/outputs/deadpool_guernica_499.png) | ![Generated Image](https://github.com/kozistr/style-transfer/blob/master/outputs/deadpool_monet_499.png) | ![Generated Image](https://github.com/kozistr/style-transfer/blob/master/outputs/deadpool_vangogh_499.png)

### Stata with wave & udnie (1024 x 679)

*conetent\style* | *wave* | *udnie*
:---: | :---: | :---: |
![Generated Image](https://github.com/kozistr/style-transfer/blob/master/contents/stata.png) | ![Generated Image](https://github.com/kozistr/style-transfer/blob/master/outputs/stata_wave_499.png) | ![Generated Image](https://github.com/kozistr/style-transfer/blob/master/outputs/stata_udnie_499.png)

## Author
HyeongChan Kim / ([@kozistr](https://kozistr.github.io), [@zer0day](http://zer0day.tistory.com))
