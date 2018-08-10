# 百度商家招牌检测比赛
----

## 运行环境：
python：3.6.3
pytorch：0.4.0
显卡	：一块 nvidia 1080ti

## 数据：
训练和测试数据就是点石竞赛的初赛数据，可以在大赛链接下载：
http://dianshi.baidu.com/gemstone/competitions/detail?raceId=17

## 训练：
训练主函数入口是tran_local.py

训练模型使用se_resnet101，并且用在imagenet上训练好的模型参数初始化，然后对所有参数进行训练，也就是不会冻结部分网络。

更多细节参考具体代码

训练得到的模型会输出在./BD_SEResNeXt101_32x4d/model/SEResNeXt101_32x4d目录下

## 预测：
函数入口在./BD_SEResNeXt101_32x4d/predict/pred.py

