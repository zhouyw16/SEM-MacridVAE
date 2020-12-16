# Readme

计76 周煜威



## 文件结构

* main.py    训练、测试、可视化的**主要代码**
* model.py    不同模型的结构，包括MultiDAE, MultiVAE, **DisenVAE**(马坚鑫), **DisenEVAE**(加入初始化), DisenSR(加入社交信息)
* data.py    读取数据集，并拆分为训练集、验证集的训练部分、验证集的测试部分、测试集的训练部分、测试集的验证部分
* RecomData    数据集的存放与预处理
  * film-trust    示例数据集
    * prep.py    任意示例数据集都有其独有的预处理，将原有数据处理为以下文件(如果存在)：
    * ratings.txt    包括：user, item, rating
    * trusts.txt    包括：trustor, trustee
    * features.txt    每一列为对应item的feature，共4096行
    * categories.txt    包括：item, cate
  * prep.py    经由每个数据集的处理后，再通过这一py文件，将以上文件进行进一步拆分，包括拆分成训练验证测试部分，包括将userid转为0至N的序号，itemid转为0至M的序号，等等。
* run    训练、测试、可视化过程的内容保存



## 代码运行

```bash
### part 1
./DecomData/film-trust
对数据集进行单独处理，处理为上述格式，film-trust为示例

### part 2
./DecomData
python prep.py <dataset>

### part 3
./
python main.py --data <dataset> --model <model> --device <cuda>


### 对film-trust数据集
./DecomData/film-trust:  python prep.py
./DecomData:             python prep.py film-trust
./:                      python main.py --data film-trust --model DisenVAE --device cuda:0
```



