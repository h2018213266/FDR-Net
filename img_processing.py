import glob
import os

import albumentations
import cv2
import numpy as np
from PIL import Image, ImageEnhance

size = 224
one_transform = albumentations.Compose([
    albumentations.Resize(size, size),
    # albumentations.GaussianBlur(blur_limit=2, sigma_limit=0, p=1.0),  # 高斯滤波
    # albumentations.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=1.0),    # CLAHE
])
def img_resize():
    path = "./dr/PDR/*.jpg"
    for i, image_path in enumerate(glob.glob(path)):
        image = cv2.imread(image_path)
        h, w, channels = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_transform = albumentations.Crop(x_min=abs(int((w - h) // 2)), y_min=0,
                                             x_max=abs(int((h + w) // 2)) if abs(int((h + w) // 2)) < w else w,
                                             y_max=h)
        # 空间变化
        image = crop_transform(image=image)['image']
        # 像素变换
        # image = one_transform(image=image)['image']
        # image_path = image_path.replace('DDR_test', 'DDR_processed')
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("图片已转换：", i)
    print("-----------done-----------")
# img_resize()

data_transform = albumentations.Compose([

        albumentations.OneOf([
            # albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4,
            #                                  value=None,
            #                                  mask_value=None, always_apply=False, p=0.5),  # 光学畸变

            # albumentations.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None,
            #                              min_width=None, fill_value=0, always_apply=False, p=0.1),  # 随机生成矩形区域
            # albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None,
            #                                 mask_value=None, always_apply=False, approximate=False, p=0.5),# 弹性变换
            # albumentations.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None,
            #                always_apply=False, p=0.5),  # 网格失真
            # albumentations.ChannelShuffle(always_apply=False, p=0.2),# 修改RGB通道
            # albumentations.InvertImg(always_apply=False, p=0.2), # 反转图像

        ]),
        albumentations.OneOf([
            albumentations.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
            # 施加摄像头传感器噪声
            # albumentations.ToGray(p=0.5),  #灰度
            # albumentations.ChannelDropout(p=0.5),# 通道丢失
            albumentations.Sharpen((1, 1), p=0.5),  # 锐化
            albumentations.Downscale(p=0.5) ,# 下采样
            # albumentations.FancyPCA(p=0.5), # 对于一张图 抽出主要特征
            # albumentations.RandomToneCurve(p=0.5),#随机色调曲线
            # albumentations.ColorJitter(p=0.5),  # 随机改变图像的亮度、对比度和饱和度
            # albumentations.OpticalDistortion(distort_limit=0.5, p =0.5),# 扭曲
            # albumentations.Superpixels(p=0.5),  # 超像素

            ]),

        albumentations.OneOf([
            albumentations.HorizontalFlip(p=0.5),  # 水平翻转
            albumentations.VerticalFlip(p=0.5),  # 垂直翻转
            # albumentations.RandomRotate90(p=0.5),  # 随机旋转图像
            # albumentations.CenterCrop(512, 512, p=0.5),
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.5),  # 亮度
            # albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 对比度
        ]),
        albumentations.OneOf([
            albumentations.MedianBlur(blur_limit=3, p=0.5),  # 中值滤波
            albumentations.Blur(blur_limit=3, p=0.5),  # 使用随机大小的内核模糊输入图像。
            albumentations.GaussianBlur(blur_limit=2, sigma_limit=0, p=0.5),  # 高斯滤波
            albumentations.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  # CLAHE
        ]),

    ])
def data_Augmentation():
    path = r"D:\DR\github\deep-learning-for-image-processing\data_set\dr4\class3\*.jpg"

    for i, image_path in enumerate(glob.glob(path)):
        print("正在处理", i+1)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = data_transform(image=image)['image']
        img_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(img_name)[0]
        re_name = name_without_ext+'-2'
        save_path = os.path.join(r'D:\DR\github\deep-learning-for-image-processing\data_set\dr4\class3', img_name)
        save_path = save_path.replace(name_without_ext, re_name)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("over done")
# data_Augmentation()

single_transform = albumentations.Compose([
    # albumentations.OneOf([
    #     albumentations.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
    #     # 施加摄像头传感器噪声
    #     albumentations.Sharpen((1, 1), p=0.5),  # 锐化
    #     albumentations.Downscale(p=0.5),  # 下采样
    # ]),
    # albumentations.MedianBlur(blur_limit=3, p=1),  # 中值滤波
    # albumentations.Blur(blur_limit=3, p=1),  # 使用随机大小的内核模糊输入图像。
    albumentations.RandomGamma(gamma_limit=(60, 120), p=1),  # 亮度
    albumentations.GaussianBlur(blur_limit=2, sigma_limit=0, p=1),  # 高斯滤波
    albumentations.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=1),  # CLAHE
    ])

def data_Augmentation_single():
    # path = r"D:\DR\github\deep-learning-for-image-processing\data_set\dr4\class3\*.jpg"
    path = r"D:\DR\DDR_processed\newlabels\DDR_1\0_images\20170504101207303.jpg"

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = single_transform(image=image)['image']
    #
    image = Image.fromarray(image)
    image_transformer = ImageEnhance.Brightness(image)
    image = image_transformer.enhance(1.4)

    img_name = os.path.basename(path)
    print(img_name)
    name_without_ext = os.path.splitext(img_name)[0]
    re_name = name_without_ext+'-4'
    save_path = os.path.join(r'D:\DR\DDR_processed\newlabels\DDR_1\augmentation', img_name)
    save_path = save_path.replace(name_without_ext, re_name)
    print(save_path)
    cv2.imwrite(save_path, cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    print("over done")
data_Augmentation_single()