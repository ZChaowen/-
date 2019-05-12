# -Task1 数据集探索 (2 days)
数据集
数据集：中、英文数据集各一份

中文数据集：THUCNews

THUCNews数据子集：https://pan.baidu.com/s/1hugrfRu 密码：qfud

英文数据集：IMDB数据集 Sentiment Analysis

IMDB数据集下载和探索
参考TensorFlow官方教程：影评文本分类  |  TensorFlow：https://tensorflow.google.cn/tutorials/keras/basic_text_classification

环境
Python 2/3 (感谢howie.hu调试Python2环境)
TensorFlow 1.3以上
numpy
scikit-learn
scipy

清华NLP组提供的THUCNews新闻文本分类数据集的一个子集（原始的数据集大约74万篇文档）。

训练使用了其中的10个分类，每个分类6500条，总共65000条新闻数据。

类别如下：

体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
数据集划分如下：

训练集: 5000*10
验证集: 500*10
测试集: 1000*10

测试时使用了混淆矩阵（recall,precision,f1_score,acc）作为评估标准 
