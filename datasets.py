# @Time : 2020-04-11 21:20
# @Author : Ben
# @Version：V 0.1
# @File : datasets.py
# @desc :数据集处理类


# from torchvision import transforms
# from PIL import Image
#
# loader = transforms.Compose([
#     # transforms.Resize(imsize),  # 缩放导入的图像
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
#
# def image_loader(image_size, image_name):
#     image = Image.open(image_name)
#     height, width = image.size
#     image = image.resize((image_size, image_size))
#     # 需要伪造的批次尺寸以适合网络的输入尺寸
#     image = loader(image).unsqueeze(0)
#     return image, height, width
#
#
# def load_image(image):
#     trans = transforms.Compose([
#         # transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     # content_image = Image.open(content_image_path)
#     # style_image = Image.open(style_image_path)
#     image = trans(image).unsqueeze(0)
#     # style_image = trans(style_image).unsqueeze(0)
#     return image
#
#
# def unload_image(image_name, image_tensor, imgHeight, imgWidth):
#     """
#     根据文件名，将tensor转化为原有格式的
#     :param image_tensor: 图片的tensor数据
#     :param imgHeight: 还原的高
#     :param imgWidth: 还原的宽
#     """
#     image = image_tensor.cpu().clone()
#     image = image.squeeze(0)
#     unloader = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((imgHeight, imgWidth))
#     ])
#     image = unloader(image)
#     image.save(image_name)
#
#
# if __name__ == '__main__':
#     content, height, width = image_loader(512, "images/input/cat.jpg")
#     print(content.shape)
#     print(height, width)


import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def image_loader(stdImgSize, image_path):
    """
    将PIL格式的图片转为tensor
    :param stdImgSize: 缩放后的图片大小
    :param image_name: 图片路径
    :return: tensor格式的图片数据，高，宽
    """
    loader = transforms.Compose([
        transforms.Resize((stdImgSize, stdImgSize)),  # scale imported image
        transforms.ToTensor()  # transform it into a torch tensor
    ])
    image = Image.open(image_path)
    height, width = image.size
    image = loader(image)
    image = image.unsqueeze(0)  # 增加batch size为1的维度
    return image, height, width


def image_unloader(image_name, image_tensor, imgHeight, imgWidth):
    """
    将tensor格式的图片转换成图片并保存
    :param image_name: 保存的图片路径
    :param image_tensor: tensor格式的图片
    :param imgHeight: 高
    :param imgWidth: 宽
    """
    unloader = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((imgWidth, imgHeight))
    ])
    image = image_tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(image_name)

