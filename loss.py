# @Time : 2020-04-12 11:32
# @Author : Ben
# @Version：V 0.1
# @File : loss.py
# @desc :编写内容损失以及风格损失


import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    """
    图片内容损失
    """

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.mse_loss = nn.MSELoss()

    def forward(self, input):
        self.loss = self.mse_loss(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):
    """
    格拉姆矩阵（图片风格对比）
    """

    def forward(self, input):
        n, c, h, w = input.size()  # a=batch size(=1)
        # c=number of feature maps
        # (h,w)=dimensions of a f. map (N=h*w)

        features = input.view(n * c, h * w)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(n * c * h * w)


class StyleLoss(nn.Module):
    """
    图片风格损失
    """

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.mse_loss = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.mse_loss(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
