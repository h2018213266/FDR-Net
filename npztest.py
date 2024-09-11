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
    # 图片路径
    path = r"D:\DR\DDR_processed\images\*.jpg"
    # path = r"D:\DR\seg\seg_done\images\*.jpg"
    # 项目中存放训练所用的npz文件路径
    path2 = r'train_npz\\'

    for i, img_path in enumerate(glob.glob(path)):
        # 读入标签
        label_path = img_path.replace('images', 'labels\\HE')   # EX HE MA SE \\ R G B Y
        # label_path = img_path.replace('images', 'labels')   # EX HE MA SE
        label_path = label_path.replace('jpg', 'tif')
        # print(label_path)
        # 图像和标签进行变换
        # 读入图像
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
    # albumentations.Resize(size2, size2),
    # albumentations.Resize(height=size2, width=size2, interpolation=cv2.INTER_AREA),
    # albumentations.LongestMaxSize(max_size=size1),
    # albumentations.CenterCrop(width=512, height=512),
})

# 去除图片黑边
def remove():
    # path = r"D:\DR\seg\images\*.jpg"
    path = r"D:\DR\DDR_test\images\*.jpg"

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


# 合并多张图片
def alltone():
    folder_paths = r"D:\DR\DDR_processed\newlabels\DDR_1\EX\*.tif" # 黄色
    for i, image1_path in enumerate(glob.glob(folder_paths)):
        image2_path = image1_path.replace('EX', 'HE')   # 红色
        image3_path = image1_path.replace('EX', 'MA')   # 绿色
        image4_path = image1_path.replace('EX', 'SE')   # 蓝色
        image_paths = {
            'EX': image1_path,
            'HE': image2_path,
            'MA': image3_path,
            'SE': image4_path
        }
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
        print('正在合并第', i + 1, "张标签")
        save_path = image1_path.replace('EX', 'Con_1')
        blended_image.save(save_path)
    print("conbined_finished")

# alltone()

# 转化为特定格式图片
def normal_Color():
    # image_path = r"D:\DR\DDR_processed\labels_color\SE\007-6686-400.tif"
    label_folder= r"D:\DR\DDR_processed\labels_color\ALL1\*.tif"
    for i, image_path in enumerate(glob.glob(label_folder)):
        image = Image.open(image_path).convert('RGBA')
        # 将图像转换为NumPy数组
        np_image = np.array(image)
        # 获取图像的宽度和高度
        width, height, _ = np_image.shape
        # 将黄色替换为4
        np_image[(np_image[:, :, 0] == 255) & (np_image[:, :, 1] == 255)] = [4, 4, 4, 255]
        # 将红色替换为1
        np_image[np_image[:, :, 0] == 255] = [1, 1, 1, 255]
        # 将绿色替换为2
        np_image[np_image[:, :, 1] == 255] = [2, 2, 2, 255]
        # 将蓝色替换为3
        np_image[np_image[:, :, 2] == 255] = [3, 3, 3, 255]
        # 创建新的Image对象
        modified_image = Image.fromarray(np_image)
        # 保存更改后的图像
        # modified_image.show()
        print('正在处理第', i + 1, "张标签")
        save_path = image_path.replace('ALL1', 'ALL2')
        modified_image.save(save_path)
    print("finished")

# normal_Color()

# 转换回原图像
def restore_Color():
    modified_folder = r"\output\predictions\*.tif"
    for i, modified_path in enumerate(glob.glob(modified_folder)):
        modified_image = Image.open(modified_path).convert('RGBA')
        np_modified_image = np.array(modified_image)
        np_restored_image = np.zeros_like(np_modified_image)
        # 将4替换回黄色
        np_restored_image[(np_modified_image[:, :, 0] == 4) & (np_modified_image[:, :, 1] == 4) & (
                    np_modified_image[:, :, 2] == 4)] = [255, 255, 0, 255]
        # 将1替换回红色
        np_restored_image[np_modified_image[:, :, 0] == 1] = [255, 0, 0, 255]
        # 将2替换回绿色
        np_restored_image[np_modified_image[:, :, 0] == 2] = [0, 255, 0, 255]
        # 将3替换回蓝色
        np_restored_image[np_modified_image[:, :, 0] == 3] = [0, 0, 255, 255]

        np_restored_image[(np_modified_image[:, :, 0] == 0) & (np_modified_image[:, :, 1] == 0) & (
                    np_modified_image[:, :, 2] ==0)] = [0, 0, 0, 255]

        restored_image = Image.fromarray(np_restored_image)
        img_name = os.path.basename(modified_path)
        # print(img_name)
        save_path = os.path.join('D:\DR\DDR_processed\labels_color\ALL5',img_name)
        # save_path = save_path.replace('png','tif')
        # name_without_ext = os.path.splitext(img_name)[0]
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
    # albumentations.GaussianBlur(blur_limit=3, sigma_limit=0, p=1.0),  # 高斯滤波
    # albumentations.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    # albumentations.HorizontalFlip(p=1),  # 水平翻转
    # albumentations.VerticalFlip(p=1),  # 垂直翻转
    # albumentations.RandomRotate90(p=1),  # 随机旋转图像
# albumentations.RandomGamma(gamma_limit=(80, 120), p=1),  # 亮度
    albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  # 对比度
    # albumentations.OneOf([
    #

    #
    # ]),
})
# 使用albumentations库对图片和对应标签作修改
def test2():
    # image_path = r"D:\DR\seg\images\007-4847-300.jpg"
    image_path = r"\data_set\dr\PDR\007-3322-200.jpg"
    label_path = r"D:\DR\seg\labels\007-4847-300.tif"
    image = cv2.imread(image_path)
    # label = cv2.imread(label_path)
    h, w, channels = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_transform = albumentations.Crop(x_min=abs(int((w - h) // 2)), y_min=0,
                                         x_max=abs(int((h + w) // 2)) if abs(int((h + w) // 2)) < w else w, y_max=h)
    image = test2_transform(image=image)['image']
    # 空间变化
    # arg = crop_transform(image=image, mask=label)
    # image = arg['image']
    # label = arg['mask']
    # plt.subplot(1, 2, 1)  # 第一行第一列
    plt.imshow(image)
    # plt.subplot(1, 2, 2)  # 第一行第二列
    # plt.imshow(label)
    plt.show()
    # cv2.imshow("original image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

# 合并两张标签图片
def conbined():
    # 打开原始图像和标签图像
    image1 = Image.open(r'D:\DR\DDR_processed\labels_color\EX\007-6093-300.tif').convert('RGBA')
    image2 = Image.open(r'D:\DR\DDR_processed\labels_color\SE\007-6093-300.tif').convert('RGBA')
    # 确保两张图片的尺寸相同
    image1 = image1.resize(image2.size)
    # 创建新的图像对象
    blended_image = Image.new("RGBA", image1.size)
    # 遍历每个像素
    for x in range(image1.width):
        for y in range(image1.height):
            # 获取两张图片对应位置的像素值
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            # 检查第二张图片是否为黑色背景
            if pixel2[0] == 0 and pixel2[1] == 0 and pixel2[2] == 0:
                # 如果第二张图片对应位置为黑色，则使用第一张图片的像素值
                blended_image.putpixel((x, y), pixel1)
            else:
                # 如果第二张图片对应位置不是黑色，则使用第二张图片的像素值
                blended_image.putpixel((x, y), pixel2)
    # 显示混合后的图像
    blended_image.show()

# conbined()

# 合并四张图片
def fourtone():
    image1 = Image.open(r'D:\DR\DDR_processed\labels_color\EX\007-6093-300.tif').convert('RGBA')
    image2 = Image.open(r'D:\DR\DDR_processed\labels_color\HE\007-6093-300.tif').convert('RGBA')
    image3 = Image.open(r'D:\DR\DDR_processed\labels_color\MA\007-6093-300.tif').convert('RGBA')
    image4 = Image.open(r'D:\DR\DDR_processed\labels_color\SE\007-6093-300.tif').convert('RGBA')
    # 确保所有图像具有相同的尺寸
    image1 = image1.resize(image2.size)
    image3 = image3.resize(image2.size)
    image4 = image4.resize(image2.size)
    # 创建新的图像对象
    blended_image = Image.new("RGBA", image1.size)
    # 遍历每个像素
    for x in range(image1.width):
        for y in range(image1.height):
            # 获取四张图片对应位置的像素值
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))
            pixel3 = image3.getpixel((x, y))
            pixel4 = image4.getpixel((x, y))
            # 判断第二张图片是否为黑色背景
            if pixel2[0] == 0 and pixel2[1] == 0 and pixel2[2] == 0:
                # 如果第二张图片对应位置为黑色，则使用第一张图片的像素值
                blended_image.putpixel((x, y), pixel1)
            else:
                # 如果第二张图片对应位置不是黑色，则使用第二张图片的像素值
                blended_image.putpixel((x, y), pixel2)
            # 判断第三张图片是否为黑色背景
            # if pixel3[0] == 0 and pixel3[1] == 0 and pixel3[2] == 0:
            #     # 如果第三张图片对应位置为黑色，则使用之前混合结果的像素值
            #     continue
            # else:
            #     # 如果第三张图片对应位置不是黑色，则混合第三张图片的像素值
            #     blended_image.putpixel((x, y), pixel3)
            # 判断第四张图片是否为黑色背景
            if pixel4[0] == 0 and pixel4[1] == 0 and pixel4[2] == 0:
                # 如果第四张图片对应位置为黑色，则使用之前混合结果的像素值
                continue
            else:
                # 如果第四张图片对应位置不是黑色，则混合第四张图片的像素值
                blended_image.putpixel((x, y), pixel4)
    # 显示混合后的图像
    blended_image.show()
# fourtone()

def foufou():
    image1 = Image.open(r'D:\DR\DDR_processed\labels_color\EX\007-6093-300.tif').convert('RGBA')    # 红色
    image2 = Image.open(r'D:\DR\DDR_processed\labels_color\HE\007-6093-300.tif').convert('RGBA')    # 蓝色
    image3 = Image.open(r'D:\DR\DDR_processed\labels_color\MA\007-6093-300.tif').convert('RGBA')    # 绿色
    image4 = Image.open(r'D:\DR\DDR_processed\labels_color\SE\007-6093-300.tif').convert('RGBA')    # 黄色

    image2.resize(image1.size)
    image3.resize(image1.size)
    image4.resize(image1.size)

    image1_arr = np.array(image1)
    image2_arr = np.array(image2)
    image3_arr = np.array(image3)
    image4_arr = np.array(image4)
    # 判断第二张图片是否为黑色背景
    is_background = (image2_arr[:, :, 0] == 0) & (image2_arr[:, :, 1] == 0) & (image2_arr[:, :, 2] == 0)
    blended_image_arr = np.where(is_background[:, :, np.newaxis], image1_arr, image2_arr)
    blended_image_arr = np.where(~is_background[:, :, np.newaxis], blended_image_arr, image3_arr)
    # blended_image_arr = np.where(~is_background[:, :, np.newaxis], blended_image_arr, image4_arr)
    # is_background = (image2_arr[:, :, 0] == 0) & (image2_arr[:, :, 1] == 0) & (image2_arr[:, :, 2] == 0)
    # blended_image_arr = np.where(is_background[:, :, np.newaxis], image1_arr, image2_arr)
    # blended_image_arr = np.where(~is_background[:, :, np.newaxis], blended_image_arr, image3_arr)
    # blended_image_arr = np.where(is_background[:, :, np.newaxis] & (blended_image_arr == image3_arr), blended_image_arr,
    #                              image4_arr)

    # 将处理后的结果转换回图像对象
    blended_image = Image.fromarray(blended_image_arr.astype(np.uint8))
    # 显示混合后的图像
    blended_image.show()

# foufou()

def ff1():
    # folder_paths = r"D:\DR\DDR_processed\labels_color\EX\*.tif"
    # for i, image1_path in enumerate(glob.glob(folder_paths)):
    #     image2_path = image1_path.replace('EX', 'HE')
    #     image3_path = image1_path.replace('EX', 'MA')
    #     image4_path = image1_path.replace('EX', 'SE')

    image_paths = {
        'EX': r'D:\DR\DDR_processed\labels_color\EX\007-6093-300.tif',
        'HE': r'D:\DR\DDR_processed\labels_color\HE\007-6093-300.tif',
        'MA': r'D:\DR\DDR_processed\labels_color\MA\007-6093-300.tif',
        'SE': r'D:\DR\DDR_processed\labels_color\SE\007-6093-300.tif'
    }

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

    blended_image.show()

# ff1()

def ff2():
    folder_paths = r"D:\DR\DDR_processed\labels_color\EX\*.tif"
    for i, image1_path in enumerate(glob.glob(folder_paths)):
        image2_path = image1_path.replace('EX', 'HE')
        image3_path = image1_path.replace('EX', 'MA')
        image4_path = image1_path.replace('EX', 'SE')

        image1 = Image.open(image1_path).convert('RGBA')
        image2 = Image.open(image2_path).convert('RGBA')
        image3 = Image.open(image3_path).convert('RGBA')
        image4 = Image.open(image4_path).convert('RGBA')
        # 创建新的图像对象
        blended_image = Image.new("RGBA", image1.size)
        # 遍历每个像素
        for x in range(image1.width):
            for y in range(image1.height):
                # 获取四张图片对应位置的像素值
                pixel1 = image1.getpixel((x, y))
                pixel2 = image2.getpixel((x, y))
                pixel3 = image3.getpixel((x, y))
                pixel4 = image4.getpixel((x, y))
                # 判断第二张图片是否为黑色背景
                if pixel2[0] == 0 and pixel2[1] == 0 and pixel2[2] == 0:
                    # 如果第二张图片对应位置为黑色，则使用第一张图片的像素值
                    blended_image.putpixel((x, y), pixel1)
                else:
                    # 如果第二张图片对应位置不是黑色，则使用第二张图片的像素值
                    blended_image.putpixel((x, y), pixel2)
                # 判断第三张图片是否为黑色背景
                if pixel3[0] == 0 and pixel3[1] == 0 and pixel3[2] == 0:
                    # 如果第三张图片对应位置为黑色，则使用之前混合结果的像素值
                    continue
                else:
                    # 如果第三张图片对应位置不是黑色，则混合第三张图片的像素值
                    blended_image.putpixel((x, y), pixel3)
                # 判断第四张图片是否为黑色背景
                if pixel4[0] == 0 and pixel4[1] == 0 and pixel4[2] == 0:
                    # 如果第四张图片对应位置为黑色，则使用之前混合结果的像素值
                    continue
                else:
                    # 如果第四张图片对应位置不是黑色，则混合第四张图片的像素值
                    blended_image.putpixel((x, y), pixel4)
        # 保存混合后的图像
        # blended_image.show()
        print('正在合并第', i + 1, "张标签")
        save_path = image1_path.replace('EX', 'ALL')
        blended_image.save(save_path)
    print("conbined_finished")

# ff2()

# 遍历四个图片文件夹
def demo1():
    folder_paths = [
        r'D:\DR\DDR_processed\labels_color\EX',
        r'D:\DR\DDR_processed\labels_color\HE',
        r'D:\DR\DDR_processed\labels_color\MA',
        r'D:\DR\DDR_processed\labels_color\SE'
    ]
    # 创建一个空列表来存储所有图像对象
    image_list = [[] for _ in range(len(folder_paths))]
    # 遍历文件夹路径列表
    for i, folder_path in enumerate(folder_paths):
        # 获取文件夹内的所有图片路径
        image_paths = folder_path + r'\*.tif'
        for image_path in glob.glob(image_paths):
        # 遍历图片路径
            image = Image.open(image_path).convert('RGBA')
            image_list[i].append(image)
        #     print(image_path)
    image_list[0][0].show()
# demo1()

#
def demo2():
    image_path = r"D:\DR\DDR_processed\labels\HE\007-6713-400.tif"
    # label = cv2.imread(path, flags=0).convert('L')
    # label = np.array(label)
    # label[label != 255] = 0
    # # 将目标像素设置为1
    # label[label == 255] = 1
    # # modified_image = Image.fromarray(label)
    # cv2.imshow("original label", label)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = Image.open(image_path).convert('L')
    # 将图像转换为NumPy数组
    label = np.array(image)
    # 将非255的像素值设为0
    label[label != 255] = 0
    # 将值为255的像素值设为1
    label[label == 255] = 1
    # 创建新的Image对象
    modified_image = Image.fromarray(label)
    # 显示修改后的图像
    modified_image.show()

# demo2()


# blend融合 两张
def mixed():
    image1 = Image.open(r'D:\DR\DDR_processed\labels_color\EX\007-6926-400.tif')
    image2 = Image.open(r'D:\DR\DDR_processed\labels_color\HE\007-6926-400.tif')
    image1 = image1.convert('RGBA')
    image2 = image2.convert("RGBA")
    img = Image.blend(image1, image2, 0.5)
    img.show()
# mixed()

def demo3():
    # def restore_Color():
    image = r"D:\DR\DDR_processed\labels_color\ALL2\007-4107-200.tif"
    # for i, modified_path in enumerate(glob.glob(modified_folder)):
    modified_image = Image.open(image).convert('RGBA')
    np_modified_image = np.array(modified_image)
    np_restored_image = np.zeros_like(np_modified_image)

    # 将4替换回黄色
    np_restored_image[(np_modified_image[:, :, 0] == 4) & (np_modified_image[:, :, 1] == 4) & (
                np_modified_image[:, :, 2] == 4)] = [255, 255, 0, 255]
    # 将1替换回红色
    np_restored_image[np_modified_image[:, :, 0] == 1] = [255, 0, 0, 255]
    # 将2替换回绿色
    np_restored_image[np_modified_image[:, :, 0] == 2] = [0, 255, 0, 255]
    # 将3替换回蓝色
    np_restored_image[np_modified_image[:, :, 0] == 3] = [0, 0, 255, 255]

    restored_image = Image.fromarray(np_restored_image)
    restored_image.show()
    #     restored_image.save(modified_path.replace('ALL2', 'ALL1'))
    #     print('正在处理第', i + 1, "张标签")
    # print("finished")
# demo3()
