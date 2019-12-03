# Fast Point R-CNN 论文分析

这是腾讯优图和香港中文大学一起搞得发表在2019 ICCV上的工作，应该是和STD是一伙人。

# Motivation
和STD差不多，都是在说现在基于点云的3D目标检测有两大类，一类是采用体素化，一类是使用原始点。体素化会损失精确的位置信息（包括高度和体素误差）,但是体素化会使原始点云规则化，从而可以发挥卷积网络的优势，使用原始点可以保留精确的位置信息，但是对于大规模场景计算能力跟不上。因此，作者设计了一种两阶段式3D目标检测方法，综合两者的优势。

# Implementation
总的来讲，整个方法分为两个部分：VoxelRPN和RefinerNet。
结构如下图所示：

<img alt="Fast Point R-CNN-1dd87852.png" src="assets/Fast Point R-CNN-1dd87852.png" width="" height="" >

## VoxelRPN
这一部分主要是将原始点云体素化，并采用卷积网络提取特征。因此，可以分成体素化和特征提取两步。

体素化过程和之前的VoxelNet，SECOND很相似，都是对三个方向进行体素化。
不同的地方在于，之前VoxelNet每个voxel内的点都是采用35个点然后使用VFE提取voxel的特征。作者认为，对于每个voxel，采用6个点，并使用一层FC(8)就足够了。（之后应该是接了一个MaxPooling，对于每个voxel都得到一个8维的特征）

除了VoxelNet这种三个方向的体素化，也有像Pillar这种体素化方式，作者认为三维体素化然后接3D卷积的方式更有利于保护物体的三维结构。

特征提取过程如下图所示：

<img alt="Fast Point R-CNN-67d5e5bb.png" src="assets/Fast Point R-CNN-67d5e5bb.png" width="" height="" >

先是采用3D卷积进行特征提取，最后生成2D特征图，并采用多尺度结构一方面获取更大的感受野，一方面保留更好的位置信息。


## RefinerNet
RefinerNet的输入是VoxelRPN输出的proposal中的点和对应的特征。特征是从VoxelRPN中concat之后的特征图中索引到的。点 $p$ 的坐标为 $(x_p, y_p)$ ，则尺寸为 $(L_F, W_F, C_F)$ 的特征图中对应的位置为 $(\lfloor \frac{x_p L_F}{L} \rfloor, \lfloor \frac{y_p W_F}{W} \rfloor)$ ，其中 $L, W$ 是原始点云栅格化后的尺寸。

第二部分输入点云选取时，是选择第一阶段预测框1.4倍大的范围内的点，并采用了注意力机制方式实现更好的特征提取。具体过程如下图所示

<img alt="Fast Point R-CNN-cd75af18.png" src="assets/Fast Point R-CNN-cd75af18.png" width="" height="" >

## 参数设置
输入点云的范围 $[0, 70.4] \times [-40, 40] \times [-3, 1]$；
体素设置大小为 $0.1 \times 0.1 \times 0.2$；
则输入网格的大小为 $800 \times 704 \times 20$ 。

预选框角度设置为4个 $(0, 45, 90, 135)$。

## 损失函数
第一阶段的损失函数与VoxelNet相同。在分类损失时，采用Focal Loss并将 $\gamma = 10$ 。并且对于负样本分类，使用OHEM（在线困难样本挖掘）

对于第一阶段IoU大约0.5的预选框，进行第二阶段的训练，第二阶段训练没有分类任务，也没有负样本。
第二阶段则采用了回归8个点的方式。（为啥要采用这种呢？精度更高？最终的结果应该也是要提供长宽高和角度，这8个点怎么计算（不一定是规则的长方体）？）

## 数据增广
数据增广策略也是分为两大类：

整个点云的增广：左右随机翻转，$[0.95, 1.05]$ 范围的随机缩放，$[-45, 45]$之间随机角度选择。

每个物体的增广： X, Y方向的随机平移 $N(0, 1)$ 和高度方向的随机平移 $(0, 0.3)$ 。$[-18, 18]$ 随机角度旋转。需要进行碰撞检测。

向原始点云中增加物体：从物体数据库中采样20个物体，放在当前点云中，每个物体在选择时需要向外扩展0.3米，以获得更好的上下文信息。

## 训练过程
8个P40，batch size 为16，使用Adam学习器，初始学习率为0.01。
两个阶段应该是分开训练。VoxelRPN训练70个epochs，在第50和第65个epoch时，学习率乘以0.1，RefinerNet再训练70个epochs，在第40，第55和第65个epoch时，学习率分别乘以0.1。


## 实验结果
在KITTI数据集上的结果
<img alt="Fast Point R-CNN-7e872785.png" src="assets/Fast Point R-CNN-7e872785.png" width="" height="" >

数据增广策略对结果的影响
<img alt="Fast Point R-CNN-4040f18b.png" src="assets/Fast Point R-CNN-4040f18b.png" width="" height="" >

第二阶段分析

<img alt="Fast Point R-CNN-c1991d41.png" src="assets/Fast Point R-CNN-c1991d41.png" width="" height="" >

两个阶段不同距离上结果的比较

<img alt="Fast Point R-CNN-b584316a.png" src="assets/Fast Point R-CNN-b584316a.png" width="" height="" >
