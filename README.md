# QuickDraw
你画“我”猜手绘图片识别
# 基本流程及关键技术
基本分为三个阶段——训练阶段（使用 Keras 框架在训练模型）；测试阶段（然后使用手绘图片运行模型）；应用阶段（加载模型，让其判断我们手绘的图片类别）
涉及到三个关键技术：
①训练数据集：quick draw
②深度学习框架：keras，tensorflow
③方法：卷积神经网络；机器学习

# 具体部署
## 收集原始数据集并处理
由于内存容量有限，我们不会使用所有类别的图像进行训练。我们仅使用数据集中的 100 个类别，每个类别的数据可以在谷歌 Colab（https://console.cloud.google.com/storage/browser/quickdrawdataset/full/numpybitmap?pli=1）上以 NumPy 数组的形式获得，数组的大小为 [N, 784]，其中 N 为某类图像的数量。我们首先下载这个数据集：
import urllib.request
def download():
base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
for c in classes:
cls_url = c.replace('_', '%20')
path = base+cls_url+'.npy'
print(path)
urllib.request.urlretrieve(path, 'data/'+c+'.npy')
由于内存限制，我们在这里将每类图像仅仅加载 4000 张。我们还将留出其中的 20% 作为测试数据。（所以一共是40万个数据信息，8万张用作训练模型）
（2）随后对数据进行预处理操作，为训练模型做准备。该模型将使用规模为 [N, 28, 28, 1] 的批处理，并且输出规模为 [N, 100] 的概率。
（3）简化图像笔画，使用resample
（4）得到保存二进制数据的Numpy位图

## 创建初步识别模型
（1）采用keras的sequentia顺序结构
（2）三层卷积层
（3）两层全连接层
![](https://raw.githubusercontent.com/BBQldf/PicGotest/master/20220306171639.png)


**3.**** 模型结构图**

| 输入层 | 卷积层conventional 16个3\*3大小的卷积核 |
| --- | --- |
|



隐含层 | 池化层 maxpooling矩阵宽高缩小一半 |
| 卷积层conventional 32个3\*3大小的卷积核 |
| 池化层 maxpooling 矩阵宽高缩小一半 |
| 卷积层conventional 64个3\*3大小的卷积核 |
| 池化层 maxpooling 矩阵高度缩小一半 |
| 展平层 flatten |
| 全连接层 输出128维数据 激活函数为relu |
| 输出层 | 全连接层 输出100维的数据 激活函数为softmax |



- **参数说明：**

#filters整数，输出空间的维度 （即卷积中滤波器的输出数量）。
 #kernel\_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
 #strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿高度和宽度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。
 #padding:(&quot;same&quot;,&quot;valid其中之一)，&quot;valid&quot;是没有填充，即对边界数据不处理，&quot;same&quot;是有填充，即保留边界处的卷积结果。

#output\_padding: 一个整数，或者 2 个整数表示的元组或列表， 指定沿输出张量的高度和宽度的填充量如果设置为 None (默认), 输出尺寸将自动推理出来。

#data\_format: 字符串， channels\_last (默认) 或 channels\_first 之一，表示输入中维度的顺序。
 #dilation\_rate: 一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。

#activation: 要使用的激活函数 。 如果不指定，则不使用激活函数

#use\_bias: 布尔值，该层是否使用偏置向量。

#kernel\_initializer: kernel 权值矩阵的初始化器

#bias\_initializer: 偏置向量的初始化器

#kernel\_regularizer: 运用到 kernel 权值矩阵的正则化

#bias\_regularizer: 运用到偏置向量的正则化函数

#activity\_regularizer: 运用到层输出（它的激活值）的正则化函数

#kernel\_constraint: 运用到 kernel 权值矩阵的约束函数。

#bias\_constraint: 运用到偏置向量的约束函数

1. 运行测试

![](https://raw.githubusercontent.com/BBQldf/PicGotest/master/20220306171956.png)

绘图功能正常，识别功能正常。改变笔刷、粗细、颜色测试

![](https://raw.githubusercontent.com/BBQldf/PicGotest/master/20220306172010.png)

1. 问题与改进

此处不考虑模型的前提下，绘图功能基本满足需求，但图像处理部分存在瑕疵。问题在于对黑色以外的颜色进行灰度处理时，灰度值的分布特征与训练集（训练集实质上是黑色笔画的转换）不符合，导致识别率的下降，如下画轮子的情况

![](https://raw.githubusercontent.com/BBQldf/PicGotest/master/20220306172015.png) ![](https://raw.githubusercontent.com/BBQldf/PicGotest/master/20220306172021.png)

左图使用了洋红色，并没有准确识别出为轮子；而右图则识别正常。

简单二值化图像数据对识别率影响很大，有一种改进方法是统计训练集（实质上是黑色笔画）的灰度特征，将其他颜色的灰度分布都调到与其一致的水平。


# 综合分析

## 综合评估

我们的模型，对于100个训练类中（共40万组数据）的数据实现了高精度预测，最后经过5轮的训练，达到了最终92.5%左右的识别率。整个模型训练过程大约耗时12分钟，处在一个可容忍的时间范围内。

![](https://raw.githubusercontent.com/BBQldf/PicGotest/master/20220306171845.png)

## 改进

实验最开始阶段，我们并没有寻找到非常适合我们实验的训练集，这很苦恼，因为最后的训练效果一直不是非常好（预测经常出错），但是在搜索相关资料的时候，发现了quick draw这个开源的数据集，并且有相关的说明文档，这才使得我们实验继续下去。

我们在实验分析中，最开始是参照&quot;猜画小歌&quot;那个项目的流程分析，然后得到了75%的正确率，其实这对于我们探索性实验已然足够，但是我们仍然需要改进。

我们在实验过程中，一开始使用了传统的梯度下降SGD优化器，5轮训练头，得到了80%的识别率，在继续训练，正确率提升幅度很小，并且在实际操作时也确实并不能很好地识别，所以我们继续学习的过程中，看一些资料，然后换了一个优化器——adam，最终5轮训练达到了92%的正确率。

最开始我们的项目设计时是没有准备界面部分的，但是在开题报告上，老师提出了相关想法，然后我们觉得这样做也确实会使我们的项目更具有吸引力，所以增加了独立的界面设计，并且设计了蜡笔、毛笔，和彩色画部分。


# 鸣谢

谢谢所有参与其中的人！谢谢老师及教辅的付出！


