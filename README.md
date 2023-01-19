<h1 align="center" href="https://arxiv.org/pdf/2203.03041.pdf">
    Highly Accurate Dichotomous Image Segmentation （ECCV 2022）
</h1>
<p align="center">
</p>
<hr>
<h2 align="center">
    IS-Net/GT-Encoder Tuning and Data Augmentation
</h2>

<div align="center">
    If you haven't read or looked into the paper or github
    please do so before go deeper..
    <br><br>
    <a href="https://arxiv.org/pdf/2203.03041.pdf" ><img width="25px" src="utils/icons/paper.gif" ></a>
    &nbsp; 
    <a href="https://github.com/xuebinqin/DIS"><img width="28px" src="utils/icons/github.gif"></a>
</div>
 
<hr>

# Data Augmentation
## Edged GT
<img src="utils/images/GT_augementation.png">

- Added BCE_Loss calculation between the output and Edged after filled GT loss calculation to aid the seperation between foreground and background.
- Got reference from [EGNett: Edge Guidance Network for Salient Object Detection](https://arxiv.org/pdf/1908.08297.pdf) which is mainly about aiding discriminate foreground better from similar color range background.

## Random Blur
<img src="utils/images/random_blur.png">

- Added blurred patch on foreground and background boundary locations to all original images to increase complexity.
- Extracted the biggest object from GT by selecting the largest contour area.
- Number of pathces can be chosen when calling function by passing down through keyword argument
- Area of patches will be randomly chosen.

```python
from src.utils.augmentation import RandomBlur

random_blur = RandomBlur()
random_blur(image="opencv original image", mask="opencv GT mask", patches=2)
```

## [Albumentation](https://albumentations.ai/) 
- Resize 1280x1280
- RandomCrop 1024x1024
- 50% Random Horizontal Flip
- 80% Random Vertical Flip
- 80% 90 degree Rotation
- 80% ElasticTransform

# Model Tuning
## GT Encoder
- In SOTA model, All feature maps below EN_2 stage, the size of output get smaller as stage goes on.
- In order to preserve detailed pixels while encoding, instead of resizing upsample, I've continuously enlarged with [convolutional transpose 2d]() to all stages until its' same size as EN_2 shape.
- Even if GT Encoder is overfitted, during Feature Synchronization with Image Segmentation Component, stages below EN_2's preservation seemed isn't suitable for pixel wise segmentation.
- 

[//]: # ()
[//]: # (- DISNET의 decoder부분에서 작아진 이미지들을 영상처리가 아닌 딥러닝으로 이미지를 키워 데이터 손실을 줄이고 GT에 가까운 side outputs들을 추출하여 loss 계산할때 큰 도움을 받는것으로 보였다.)

[//]: # (- 보다 나은 독해를 위해 용어 정리 해 두었다.)

[//]: # (- convolution_transpose_2d를 단순히 deconvolutional_upsample, 또는 deconv_upsample로 표현하겠다.)

[//]: # ()
[//]: # ()
[//]: # (- GTNet 또한 DISNet과 동일한 조건을 맞춰주기 위해 stage2 낮은 stage들을 stage2 크기로 deconv_upsample로 키워서 over-fitting을 시켰다.)

[//]: # (- GTNet은 batch_size 14에 epoch 487번을 돌아 validation_loss를 0.20을 달성해 학습을 종료 시켰다.)

[//]: # (- visualization을 통해 섬세한 side_outputs들이 나왔다 &#40;추후 이미지를 공개할 예정이다&#41;.)

[//]: # ()
[//]: # (# DISNET Deconvolutional Stage Test)

[//]: # (- deconv_upsampled는 batch_size 14에 epoch 339번을 돌았다.)

[//]: # ()
[//]: # (|ORIGINAL_IMAGE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DECONV_TO_D2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ORIGINAL_ISNET|)

[//]: # (|------------|)

[//]: # (|![d5d6_vs_d2]&#40;sample_images/d2up_isnet-pretrained.png&#41;)

[//]: # ()
[//]: # (# PREREQUISITE)

[//]: # (### 아나콘다 환경설정)

[//]: # (- 아나콘다 환경 라이브러리 설치)

[//]: # (```sh)

[//]: # (conda env create --file pytorch_env.yml )

[//]: # (```)

[//]: # (- 아나콘다 환경설정에 대한 자세한 설명은 [가상환경 그대로 옮기기]&#40;https://velog.io/@sheoyonj/Conda-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EA%B7%B8%EB%8C%80%EB%A1%9C-%EC%98%AE%EA%B8%B0%EA%B8%B0&#41; 참조)

[//]: # ()
[//]: # (# RUN)

[//]: # (```sh)

[//]: # (python )

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (# References)

[//]: # (- DISNET: [xuebinqin/DIS]&#40;https://github.com/xuebinqin/DIS&#41;)

[//]: # (- U2NET: [xuebinqin/U-2-NET]&#40;https://github.com/xuebinqin/U-2-Net&#41;)

[//]: # (- EGNET: [JXingZhao/EGNet]&#40;https://github.com/JXingZhao/EGNet&#41;)
