# paper-2018-HAASD
HAASD: A dataset of Household Appliances Abnormal Sound Detection - paper replication data
此项目是使用深度学习对洗衣机等家电异音经行检测，从而实现自动对故障家电与正常家电经行分类的目的。项目内容如下：

一、使用工具
1、程序语言：python 
2、IDE：pycharm
3、深度学习框架：Tensorflow
4、matlab

二、数据集
t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz、train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz四个文件分别是测试集、测试集的标签集、训练集、训练集的标签集
tmp：                       该文件夹里存放的是数据集，在创建项目后，请将该文件夹存放入项目所在磁盘的根目录中
此文件夹中的两个.m文件是处理原始数据所用程序，一个是生成训练集及其标签，另一个是生成测试集及其标签，将原始数据处理生成二进制文件

三、代码
swallowsound:               该文件夹里存放洗衣机异音检测项目的主要程序，在pycharm中创建一个新项目并且创建一个名为swallowsound的包，把该文件夹中所有文件放入创建的这个包所在文件夹中即可运行noise_deep_train.py这个已经训练好的模型，输出结果是一个准确率
swallowsound_input_data.py：该程序是用来导入训练集，单独封装成一个模块可供其他模块调用，提高代码的重复使用率
noise_softmax.py：          该程序是使用softmax作为分类器对数据经行了简单分类
noise_deep_3cnn.py：       该程序是使用三层卷积核卷积神经网络对数据经行训练并给出预测结果
logs：                      该文件夹里存放了训练过程中保存的可视化文件，在terminal中运行：tensorboard  --logdir=D:\pycharm\swallowsound\swallowsound\logs\ 后可按照提示即可在浏览器中看到可视化内容
mynet：                     该文件夹中存放的是训练时存放的数据，在训练程序中断后可供下次接着本次训练结果继续训练

四、评估
预测结果的准确率最高达到85%


项目GitHub链接：https://github.com/JYongSmile/swallowsoundml.git
如果有什么疑问或者运行程序出现什么问题可以微信联系，欢迎交流沟通，共同进步！	邮箱：s.barry1994@foxmail.com  CSDN博客：http://blog.csdn.net/barry_j
