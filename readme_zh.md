# Readme



## 文件结构

* main.py    训练、测试、可视化的**主要代码**
* model.py    不同模型的结构，包括MultiDAE, MultiVAE, **DisenVAE**(马坚鑫，即MacridVAE), **DisenEVAE**(加入初始化，即SEM-MacridVAE)
* data.py    读取数据集，并拆分为训练集、验证集的训练部分、验证集的测试部分、测试集的训练部分、测试集的验证部分
* RecomData    数据集的存放与预处理
  * ml-latest-small    示例数据集
    * prep.py    任意示例数据集都有其独有的预处理，将原有数据处理为以下文件(如果存在)：
    * ratings.txt    包括：user, item, rating
    * embed.npy    图片的特征向量
  * prep.py    经由每个数据集的处理后，再通过这一py文件，将以上文件进行进一步拆分，包括拆分成训练验证测试部分，包括将userid转为0至N的序号，itemid转为0至M的序号，等等。
* run    训练、测试、可视化过程的内容保存



## 代码运行

```bash
### part 1
./DecomData/ml-latest-small
对数据集进行单独处理，处理为上述格式，ml-latest-small为示例

### part 2
./DecomData
python prep.py <dataset>

### part 3
./
python main.py --data <dataset> --model <model> --device <cuda>


### 对ml-latest-small数据集
./DecomData/ml-latest-small:  python prep.py
./DecomData:                  python prep.py ml-latest-small
./:                           python main.py --data ml-latest-small--model DisenEVAE --device cuda:0
```



