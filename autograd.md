# 前言
希望在尽量短的篇幅，讲清楚pytorch的主要特性，能够帮助大家快速上手，亦是个人的学习笔记，欢迎大家批评指证。因为英语太烂，所以中文版先行。
更详尽的内容参见[官网](https://pytorch.org/docs/stable/), 虽稍显啰嗦，但胜在内容翔实。
# 环境搭建
因为python出了名的版本“魔咒”，以及扶不起的包管理器即便有了`conda`和`pipenv`的加持，一劳永逸的方法还是用docker创建一个完全clean的环境，莫要自找麻烦。


## Docker
主要目的是为了快速上手pytorch，所以使用了非cuda版本（_官方的cuda版本有bug，需要自己fix_），也没刻意追求尽量小的docker size，所以选用了Ubuntu：16.04做为基层，以尽量少折腾为目的。_如果你很在意docker的大小，可以试试[这个](https://hub.docker.com/r/petronetto/pytorch-alpine/~/dockerfile/)_。所有相关事例，都打包在image里，为了方便大家试验。
```sh
docker run -it -p 8888:8888  -v $PWD:/notebooks arthursjiang/ai_learn
open http://localhost:8888
```

## Host(废弃, 自求多福)
~~- Install [miniconda](https://conda.io/miniconda.html)~~

~~- Install [pytorch](https://pytorch.org/features)~~
# 自动求导(Autograd)机制
Tensor通过`requires_grad`变量启用或禁用子图的自动求导。当所有输入都不需要求导时，对应输出也不需要求导；与输出相关的一个输入需要求导时，输出也需要求导。当你需要使用预先训练好的模型作为你网络的一部分时，可以利用`requires_grad`, 不对该部分进行更新。
自动求导基于数据操作生成一张从输入（leaf）到输出（root）的有向无环图，再基于链式规则，从输出（root）到输入（leaf）计算导数。每次迭代，图结构都会重新基于该次迭代的数据操作重新生成。
```python
x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn(5, 5, requires_grad=True)

# False, because requires_grad is False by default
print(x.requires_grad)
a = x + y
# False, because both x.requires_grad and y.requires_grad are False
print(a.requires_grad)
b = x + z
# True, because z.requires_grad is True
print(b.requires_grad)
```
## TODO more about in-place operation, and autograd mechanics

# 张量（Tensor）
`torch.Tensor`是用来存储单一数据类型的多维矩阵。torch在CPU和GPU上各支持8种数据类型。(`torch.float16, torch.float32, torch.float64, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64`)

`torch.tensor()`会引起数据拷贝，如果已有一个numpy的数组，可以用`torch.as_tensor()`避免数据拷贝。
```python
# Create tensor
x = torch.tensor([[1., -1.], [1., -1.]])
print(x.size(), x.dtype, x.device)
y = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
print(y.size(), y.dtype, y.device)
z = torch.zeros([2, 4], dtype=torch.int32)
print(z.size(), z.dtype, z.device)

# Access tensor
print(x[0][1])
print(y[0][:2])

# Get python number from a single value tensor
print(y[0][2])
print(y[0][2].item())
```

