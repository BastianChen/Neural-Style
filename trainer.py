# @Time : 2020-04-11 21:21 
# @Author : Ben 
# @Version：V 0.1
# @File : trainer.py
# @desc :结合BP算法实现风格迁移

from datasets import *
from torchvision import models
from loss import *
import copy
import os


class Trainer:
    def __init__(self, content_path, style_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = models.vgg19(pretrained=True).features.to(self.device).eval()
        # 所需的深度层以计算样式/内容损失：
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_image, self.height, self.width = image_loader(512, content_path)
        self.style_image, _, _ = image_loader(512, style_path)
        self.style_image, self.content_image = self.style_image.to(self.device), self.content_image.to(self.device)
        self.content_weight = 1
        self.style_weight = 500

    def get_input_param_optimizer(self, input_img):
        '''
        参数优化器搭建
        Gradient descent
        '''
        # this line to show that input is a parameter that requires a gradient
        input_param = nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([input_param])
        return input_param, optimizer

    def get_loss(self):
        cnn = copy.deepcopy(self.net)
        gram = GramMatrix().to(self.device)

        # 只是为了获得对内容/样式的可迭代访问或列表
        # losses
        content_losses = []
        style_losses = []

        # 假设cnn是nn.Sequential，那么我们创建一个新的nn.Sequential
        # 放入应该顺序激活的模块
        model = nn.Sequential()

        i = 0  # 每当转换时就增加
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # 旧版本与我们在下面插入的ContentLoss和StyleLoss不能很好地配合使用。
                # 因此，我们在这里替换为不适当的。
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # 增加内容损失:
                # print(self.content_image)
                target = model(self.content_image).clone()
                content_loss = ContentLoss(target, self.content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # 增加样式损失:
                target_feature = model(self.style_image).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, self.style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        # 现在我们在最后一次内容和样式丢失后修剪图层
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def transfer(self, image_name, num_steps=350):
        input_img = self.content_image.clone()
        # 如果要使用白噪声，请取消注释以下行：
        # input_img = torch.randn(self.content_image.data.size(), device=self.device)
        """Run the style transfer."""
        print(f'Building the style transfer model for {image_name}')
        print(f'Style image is {image_name}')
        model, style_losses, content_losses = self.get_loss()

        print('Optimizing..')
        input_param, optimizer = self.get_input_param_optimizer(input_img)
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # 更正更新后的输入图像的值
                input_param.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_param)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.backward()
                for cl in content_losses:
                    content_score += cl.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    image_unloader(os.path.join("images/output/", image_name), input_param.data, self.height, self.width)
                return style_score + content_score

            optimizer.step(closure)

        # 最后更正...
        input_param.data.clamp_(0, 1)


if __name__ == '__main__':
    style_image_list = os.listdir("images/style")
    for image_name in style_image_list:
        trainer = Trainer("images/content/content.jpg", os.path.join("images/style/", image_name))
        trainer.transfer(image_name)
