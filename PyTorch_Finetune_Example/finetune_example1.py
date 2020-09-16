from torchvision import models
from torch import nn
from torch import optim

resnet_model = models.resnet18(pretrained=True) 
# pretrained 设置为 True，会自动下载模型 所对应权重，并加载到模型中
# 也可以自己下载 权重，然后 load 到 模型中，源码中有 权重的地址。

# 假设 我们的 分类任务只需要 分 100 类，那么我们应该做的是
# 1. 查看 resnet 的源码
# 2. 看最后一层的 名字是啥 （在 resnet 里是 self.fc = nn.Linear(512 * block.expansion, num_classes)）
# 3. 在外面替换掉这个层
resnet_model.fc= nn.Linear(in_features=..., out_features=100)

# 这样就 哦了，修改后的模型除了输出层的参数是 随机初始化的，其他层都是用预训练的参数初始化的。

# 如果只想训练 最后一层的话，应该做的是：
# 1. 将其它层的参数 requires_grad 设置为 False
# 2. 构建一个 optimizer， optimizer 管理的参数只有最后一层的参数
# 3. 然后 backward， step 就可以了

# 这一步可以节省大量的时间，因为多数的参数不需要计算梯度
for para in list(resnet_model.parameters())[:-1]:
    para.requires_grad=False 

optimizer = optim.SGD(params=[resnet_model.fc.weight, resnet_model.fc.bias], lr=1e-3)