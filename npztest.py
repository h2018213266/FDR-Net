import os
import numpy as np
import cv2
import glob
import albumentations
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
# pytorch模型接受的输入是Tensor格式，而albumentations增强后的数据格式是numpy，添加一个numpy转为Tensor的函数即可
# images
size2 = 224
one_transform = albumentations.Compose([
    # albumentations.GaussianBlur(blur_limit=2, sigma_limit=0, p=1.0),  # 高斯滤波
    albumentations.OneOf([
        albumentations.RandomGamma(gamma_limit=(80, 120), p=0.5),  # 亮度
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 对比度
    ]),
        albumentations.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),    # CLAHE
])
# images＆lables
two_transform = albumentations.Compose({
    # albumentations.Crop(x_min=abs(int((w - h) // 2)), y_min=0,
    #                     x_max=abs(int((h + w) // 2)) if abs(int((h + w) // 2)) < w else w, y_max=h),
    albumentations.Resize(size2,size2),

    albumentations.OneOf([
        albumentations.HorizontalFlip(p=0.5),  # 水平翻转
        albumentations.VerticalFlip(p=0.5),  # 垂直翻转
        albumentations.RandomRotate90(p=0.5),  # 随机旋转图像
    ]),

})
# 将图像和标签转化成.npz文件(二值图像)
def npz():


    path2 = r'train_npz\\'

    for i, img_path in enumerate(glob.glob(path)):
        # 读入标签
        label_path = img_path.replace('images', 'labels\\HE')   
        label_path = label_path.replace('jpg', 'tif')

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, flags=0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        image = one_transform(image=image)['image']

        arg = two_transform(image=image, mask=label)
        image = arg["image"]
        label = arg["mask"]
        # 将非目标像素设置为0
        label[label != 255] = 0
        # 将目标像素设置为1
        label[label == 255] = 1
        # 获取图像名
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]
        # 保存npz
        np.savez(path2 + str(name_without_ext), image=image, label=label)
        print(i+1, '----', name_without_ext,"已转换到.NPZ")

    # 加载npz文件
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']
    print('------npz_finished-------')

# npz()

# 生成npz文件对应的txt文件
def write_name():
    # npz文件路径 训练
    files = glob.glob(r'\train_npz\*.npz')
    f = open(r'\lists\lists_Synapse\train.txt','w')
    # 测试
    for i in files:
        name = i.split('\\')[-1]
        # print(name)   007-6926-400.npz
        name = name[:-4] + '\n'
        # print(name)   007-6926-400
        f.write(name)
    print("------write_finished-------")

# write_name()

w, h = 0, 0
test_transform = albumentations.Compose({
    # crop_transform,
    albumentations.Crop(x_min=abs(int((w - h) // 2)), y_min=0,
                        x_max=abs(int((h + w) // 2)) if abs(int((h + w) // 2)) < w else w, y_max=h),

})

# 去除图片黑边
def remove():

    for i, image_path in enumerate(glob.glob(path)):
        label_path = image_path.replace('images', 'labels\\SE') #EX HE MA SE
        label_path = label_path.replace('jpg', 'tif')
        # print("image_path:",image_path,"--- label_path:",label_path)
        image = cv2.imread(image_path)
        mask = cv2.imread(label_path)
        h, w, channels = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_transform = albumentations.Crop(x_min=abs(int((w - h) // 2)), y_min=0,
                                             x_max=abs(int((h + w) // 2)) if abs(int((h + w) // 2)) < w else w, y_max=h)
        # 空间变化
        arg = crop_transform(image=image, mask=mask)
        image = arg['image']
        mask = arg['mask']
        # 像素变换
        # image = one_transform(image=image)['image']

        image_path = image_path.replace('DDR_test', 'DDR_processed')
        label_path = label_path.replace('DDR_test', 'DDR_processed')

        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(label_path, mask)
        print("图片已转换：", i)
    print("-----------done-----------")

# remove()

# 处理多张病变图片 修改标签颜色 转移到新文件夹
def change_all():
    # folder_path = r"D:\DR\DDR_processed\labels\EX\*.tif"    #MA（绿） HE（红） SE（蓝） EX（黄）
    folder_path = r"D:\DR\DDR_processed\labeled\SE\predictions\*.png"    #MA（绿） HE（红） SE（蓝） EX（黄）
    for i, image_path in enumerate(glob.glob(folder_path)):
        print("正在处理第", i+1, '张')
        # 读取标签图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 定义原始颜色和目标颜色
        original_color = (0, 255, 0)  # 原始颜色 BGR
        target_color = (255, 0, 0)  # 目标颜色 BGR(R G B Y)
        # 将原始颜色替换为目标颜色
        mask = np.all(image == original_color, axis=-1)
        # 将原始颜色替换为目标颜色
        image[mask] = target_color
        # 保存修改后的图像到同一路径下
        save_path = image_path.replace('labeled', 'newlabels')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存路径存在
        cv2.imwrite(save_path, image)
        # 保存修改后的图片
    print("change_all finished")

# change_all()

def alltone():

        images = {}
        resized_images = {}
        for key, path in image_paths.items():
            images[key] = Image.open(path).convert('RGBA')
        size = images['HE'].size
        for key, image in images.items():
            resized_images[key] = image.resize(size)
        blended_image = Image.new("RGBA", size)
        for x in range(size[0]):
            for y in range(size[1]):
                pixel_values = [resized_images[key].getpixel((x, y)) for key in resized_images]
                if pixel_values[1] == (0, 0, 0, 255):
                    blended_image.putpixel((x, y), pixel_values[0])
                else:
                    blended_image.putpixel((x, y), pixel_values[1])
                for pixel in pixel_values[2:]:
                    if pixel != (0, 0, 0, 255):
                        blended_image.putpixel((x, y), pixel)
                        break
    print("conbined_finished")


# 转化为特定格式图片
def normal_Color():
    for i, image_path in enumerate(glob.glob(label_folder)):
        image = Image.open(image_path).convert('RGBA')
        np_image = np.array(image)


        np_image[(np_image[:, :, 0] == 255) & (np_image[:, :, 1] == 255)] = [4, 4, 4, 255]
        np_image[np_image[:, :, 0] == 255] = [1, 1, 1, 255]
        np_image[np_image[:, :, 1] == 255] = [2, 2, 2, 255]
        np_image[np_image[:, :, 2] == 255] = [3, 3, 3, 255]
        modified_image = Image.fromarray(np_image)

        print('正在处理第', i + 1, "张标签")
        save_path = image_path.replace('ALL1', 'ALL2')
        modified_image.save(save_path)
    print("finished")

# normal_Color()


def restore_Color():
    modified_folder = r"\output\predictions\*.tif"
    for i, modified_path in enumerate(glob.glob(modified_folder)):
        modified_image = Image.open(modified_path).convert('RGBA')
        np_modified_image = np.array(modified_image)
        np_restored_image = np.zeros_like(np_modified_image)

        np_restored_image[(np_modified_image[:, :, 0] == 4) & (np_modified_image[:, :, 1] == 4) & (
                    np_modified_image[:, :, 2] == 4)] = [255, 255, 0, 255]

        np_restored_image[np_modified_image[:, :, 0] == 1] = [255, 0, 0, 255]
        np_restored_image[np_modified_image[:, :, 0] == 2] = [0, 255, 0, 255]

        np_restored_image[np_modified_image[:, :, 0] == 3] = [0, 0, 255, 255]

        np_restored_image[(np_modified_image[:, :, 0] == 0) & (np_modified_image[:, :, 1] == 0) & (
                    np_modified_image[:, :, 2] ==0)] = [0, 0, 0, 255]

        restored_image = Image.fromarray(np_restored_image)
        img_name = os.path.basename(modified_path)
        save_path = os.path.join(img_name)
        restored_image.save(save_path)
        print('正在处理第', i + 1, "张标签")
    print("finished")

# restore_Color()

# npz2 product（RGB图像）
def npz2():
    # 图片路径
    path = r"processed\images\*.jpg"

    path2 = r'\data\Synapse\test_vol_h5\\'
    for i, img_path in enumerate(glob.glob(path)):
        label_path = img_path.replace('images', 'labels_color\\ALL2')  # EX HE MA SE
        label_path = label_path.replace('jpg', 'tif')
        # print(label_path)
        image = cv2.imread(img_path)
        label = cv2.imread(label_path, flags=0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = one_transform(image=image)['image']
        arg = two_transform(image=image, mask=label)
        image = arg["image"]
        label = arg["mask"]
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]
        # 保存npz
        np.savez(path2 + str(name_without_ext), image=image, label=label)
        print('图片', i + 1, '----', name_without_ext, "已转换到.NPZ")
    print('------npz_finished-------')
# npz2()
# write_name()

test2_transform = albumentations.Compose({

    albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  # 对比度
    # albumentations.OneOf([
    #

    #
    # ]),
})
def test2():
    image = cv2.imread(image_path)

    h, w, channels = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_transform = albumentations.Crop(x_min=abs(int((w - h) // 2)), y_min=0,
                                         x_max=abs(int((h + w) // 2)) if abs(int((h + w) // 2)) < w else w, y_max=h)
    image = test2_transform(image=image)['image']
    plt.imshow(image)

    plt.show()

# test2()

# 修改标签颜色:单张
def change_Color():
    # 读取标签图片
    image_path = r"D:\DR\seg\labels\007-6894-400.tif"
    image = Image.open(image_path)
    # 定义原始颜色和目标颜色
    original_color = (0, 0, 0)  # 原始颜色（绿色）RGB
    target_color = (255, 0, 0)  # 目标颜色
    # 将原始颜色替换为目标颜色
    data = image.getdata()
    new_data = []
    for item in data:
        # 判断像素颜色是否为原始颜色
        if item != original_color:
            new_data.append(item)
        else:
            new_data.append(target_color)
    # 创建新的图片对象
    new_image = Image.new('RGB', image.size)
    new_image.putdata(new_data)
    # 保存修改后的图片
    # new_image.save('modified_image.png')
    new_image.show()

# change_Color()

def conbined():

    image1 = image1.resize(image2.size)
    blended_image = Image.new("RGBA", image1.size)

    for x in range(image1.width):
        for y in range(image1.height):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            if pixel2[0] == 0 and pixel2[1] == 0 and pixel2[2] == 0:
                # 如果第二张图片对应位置为黑色，则使用第一张图片的像素值
                blended_image.putpixel((x, y), pixel1)
            else:
                # 如果第二张图片对应位置不是黑色，则使用第二张图片的像素值
                blended_image.putpixel((x, y), pixel2)
    # 显示混合后的图像
    blended_image.show()

# conbined()

