import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from copy import deepcopy

# Gram矩阵计算
def gram_matrix(input):
    a, b, c, d = input.size()  # 获取输入张量的维度
    features = input.view(a * b, c * d)  # 将输入展平成矩阵
    G = torch.mm(features, features.t())  # 计算Gram矩阵
    return G

# 内容损失
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # target是预先计算好的特征图，且不需要计算梯度

    def forward(self, input):
        self.loss = torch.sum((input - self.target) ** 2) / 2.0  # 内容损失公式
        return input

# 风格损失
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()  # 计算Gram矩阵

    def forward(self, input):
        G = gram_matrix(input)  # 计算输入的Gram矩阵
        self.loss = torch.sum((G - self.target) ** 2) / (4.0 * input.size(1) ** 2 * input.size(2) * input.size(3))  # 风格损失公式
        return input

# 图像预处理与后处理
class ImageCoder:
    def __init__(self, image_size, device):
        self.device = device
        self.preproc = transforms.Compose([
            transforms.Resize(image_size),  # 改变图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1]),
            transforms.Lambda(lambda x: x.mul_(255))
        ])
        self.postproc = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1. / 255)),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
        ])
        self.to_image = transforms.ToPILImage()

    def encode(self, image_path):
        image = Image.open(image_path)
        image = self.preproc(image)
        image = image.unsqueeze(0)  # 增加batch维度
        return image.to(self.device, torch.float)

    def decode(self, image):
        image = image.cpu().clone()
        image = image.squeeze(0)  # 移除batch维度
        image = self.postproc(image)
        image = image.clamp(0, 1)  # 保证像素值在0到1之间
        return self.to_image(image)

# 目标VGG19模型
class Model:
    def __init__(self, device, image_size):
        cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        self.cnn = cnn  # 获取预训练的VGG19卷积神经网络
        self.device = device
        self.content_losses = []
        self.style_losses = []
        self.image_proc = ImageCoder(image_size, device)

    def run(self, content_image_path, style_image_path, output_image_path):
        content_image = self.image_proc.encode(content_image_path)
        style_image = self.image_proc.encode(style_image_path)
        output_image = content_image.clone().requires_grad_(True)  # 确保output_image是叶子张量
        self._build(content_image, style_image)  # 建立损失函数
        output_image = self._transfer(output_image, content_image, style_image)  # 进行最优化
        self.image_proc.decode(output_image).save(output_image_path)  # 保存图像
        return output_image_path

    def _build(self, content_image, style_image):
        self.model = nn.Sequential()
        block_idx = 1
        conv_idx = 1
        # 逐层遍历VGG19，取用需要的卷积层
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}_{}'.format(block_idx, conv_idx)
                conv_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}_{}'.format(block_idx, conv_idx)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(block_idx)
                block_idx += 1
                conv_idx = 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(block_idx)
            else:
                raise Exception("invalid layer")
            self.model.add_module(name, layer)

            # 添加内容损失函数
            if name in content_layers:
                target = self.model(content_image).detach()  # 获取目标内容特征图
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}_{}".format(block_idx, conv_idx), content_loss)
                self.content_losses.append(content_loss)

            # 添加风格损失函数
            if name in style_layers:
                target_feature = self.model(style_image).detach()  # 获取目标风格特征图
                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}_{}".format(block_idx, conv_idx), style_loss)
                self.style_losses.append(style_loss)

        # 留下有用的部分
        i = 0
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break
        self.model = self.model[:(i + 1)]

    def _transfer(self, output_image, content_image, style_image):
        optimizer = torch.optim.LBFGS([output_image])
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            def closure():
                optimizer.zero_grad()
                self.model(output_image)

                style_score = 0
                content_score = 0
                for sl, sw in zip(self.style_losses, style_weights):
                    style_score += sl.loss * sw
                for cl, cw in zip(self.content_losses, content_weights):
                    content_score += cl.loss * cw
                loss = style_score + content_score
                loss.backward()
                
                # 修改进度计算逻辑
                progress = min(100.0, (run[0] / num_steps) * 100)  # 确保不超过100%
                print(f"PROGRESS:{progress:.1f}")  # 减少小数位数
                
                run[0] += 1
                return loss
                
            optimizer.step(closure)
        return output_image

# 设定参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 512  # 增大图像分辨率，提升图像清晰度
content_layers = ['conv_4_2']  # 内容损失函数使用的卷积层
style_layers = ['conv_1_1', 'conv2_1', 'conv_3_1', 'conv_4_1', 'conv5_1']  # 风格损失函数使用的卷积层
content_weights = [5]  # 增加内容损失函数的权重
style_weights = [1e3, 1e3, 1e3, 1e3, 1e3]  # 保持风格损失函数的权重
num_steps = 500  # 增加优化步数，提升图像质量

# 运行模型
model = Model(device, image_size)

# 使用固定的文件路径
content_image_path = 'uploads/content.jpg'  # 上传的内容图片
style_image_path = 'uploads/style.jpg'      # 复制的风格图片
output_image_path = 'uploads/output.jpg'    # 输出图片

# 进行风格迁移并保存图像
output_image_path = model.run(content_image_path, style_image_path, output_image_path)
print(f"输出图像已保存至：{output_image_path}")
