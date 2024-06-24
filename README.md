# ECLV

我们的NLPCC 2024论文《ECLV：An Emotion Cause Enhanced Model With Latent Variable for Emotional Support Conversation》的代码:

请注意，我们的模型代码基于由Facebook AI Research开发的一个序列到序列模型工具包Fairseq，它可以快速构建、训练和部署各种序列到序列模型。
并支持多种训练和推理技术。我们的数据预处理，模型微调和推理都使用了Fairseq包。

### 依赖

- python==3.8.19
- torch==1.8.0+cu111
- fairseq==0.9.0
- tensorboardX==2.2
- pytorch_transformers
- scikit-learn==1.3.2
- nltk==3.8.1

```shell
sudo apt install default-jdk
curl https://install.meteor.com/ | sh

pip install -r requirements.txt
```

### 预训练模型

我们将以下检查点用于预训练模型[DialogVED](https://aclanthology.org/2022.acl-long.333/)，如论文中所述。
下载预训练的检查点，并将其放在路径`/ECLV/models`中。

- [DialogVED-VAE-Standard](https://drive.google.com/file/d/1EucujAl8vXyrEDAyAb0SeLovzIX2_9tn/view?usp=sharing)
- [DialogVED-VAE-Large](https://drive.google.com/file/d/1GLMrNAc2YEPJ-eiRcbFHGP0XGzAwKikM/view?usp=sharing)
- [DialogVED-Seq2Seq](https://drive.google.com/file/d/1xiRMBPeaIUvKFbnKrf7etXPVyU1C1x56/view?usp=sharing)

**注意**: 
我们在论文中使用的是DialogVED VAE Large。其中DialogVED VAE Standard的隐变量大小为32， DialogVED VAE Large具有大小为64的隐变量。
而DialogVED-Seq2Seq没有隐变量， 它是一个纯seq2seq模型，具有与DialogVED相同的训练设置。在响应多样性不那么重要的情况下，它可能会表现得更好。

### 在数据集上微调

#### 数据准备

我们在[ESConv](https://github.com/thu-coai/Emotional-Support-Conversation)上微调ECLV。

此项目中的ESConv数据集的原始数据路径为：
`/ECLV/data/finetune/esconv/original_data`

#### 预处理

```shell
python preprocess/process.py
```

#### 二值化

```shell
bash preprocess/binarize.sh
```

#### 微调

脚本`train.sh`有两种参数，分别为`p`和`t`。

- `p`: pretrained model **p**ath
- `t`: pretrained model **t**ype (`dialogved_standard`, `dialogved_large` or `dialogved_seq2seq`)

**注意**: 根据一些开发者的反馈，如果你的设备的GPU内存很小，无法支持默认的批次大小，请适当降低学习率，否则将无法正常收敛。
或者，您可以适当地减小`train.sh`中的参数--max-tokens和--max-sentences的大小。

```shell
bash train.sh -p models/dialogved_large.pt -t dialogved_large
```

#### 推理

脚本`infer.sh`有两种参数, 分别为`d`和`s`。

- `d`: fine-tuned **d**ataset (`esconv`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)

```shell
bash infer.sh -d esconv -s sampling
```

#### 评估

脚本`eval.sh`仅有一种参数`d`。

- `d`: fine-tuned **d**ataset (`esconv`)

```shell
bash eval.sh -d esconv
```

