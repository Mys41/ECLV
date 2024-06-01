# ECLV

Code for our NLPCC 2024 paper:  
ECLV：An Emotion Cause Enhanced Model With Latent Variable for Emotional Support Conversation

### Requirements

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

### Pre-trained Models

We use the following checkpoints for pre-trained models as described in the paper of 
[DialogVED](https://aclanthology.org/2022.acl-long.333/). Download the pre-trained checkpoint 
and set the `load-from-pretrained-model` parameter in the fine-tuning running command.  

- [DialogVED-VAE-Standard](https://drive.google.com/file/d/1EucujAl8vXyrEDAyAb0SeLovzIX2_9tn/view?usp=sharing)
- [DialogVED-VAE-Large](https://drive.google.com/file/d/1GLMrNAc2YEPJ-eiRcbFHGP0XGzAwKikM/view?usp=sharing)
- [DialogVED-Seq2Seq](https://drive.google.com/file/d/1xiRMBPeaIUvKFbnKrf7etXPVyU1C1x56/view?usp=sharing)

**Note**: What we used in the paper is DialogVED-VAE-Large. DialogVED-VAE-Standard has a size of latent size 32, 
where DialogVED-VAE-Large has a size of latent size 64. DialogVED-Seq2Seq has no latent variable, 
it's a pure seq2seq model with the same training setting like DialogVED. 
It may perform better in scenarios where diversity of responses is less important.   

### Fine-tuning on Your Dataset

#### Data preparation

We finetune ECLV on [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation).

The original data path of the ESConv dataset in this project is: 
/ECLV/data/finetune/esconv/original_data

#### Preprocess

```shell
python preprocess/process.py
```

#### Binarization

```shell
bash preprocess/binarize.sh
```

#### Training

the script `train.sh` has three parameters, namely `p` and `t`.

- `p`: pretrained model **p**ath
- `t`: pretrained model **t**ype (`dialogved_standard`, `dialogved_large` or `dialogved_seq2seq`)

**Note**: According to the feedback of some developers, if the GPU memory of your device is small and cannot support
the default batch size, please reduce the learning rate appropriately, or it will not converge normally.
Alternatively, you can reduce the size of --max-tokens and --max-sentences appropriately.

```shell
bash train.sh -p models/dialogved_large.pt -t dialogved_large
```

#### Inference

the script `infer.sh` has two parameters, namely `d` and `s`.

- `d`: fine-tuned **d**ataset (`esconv`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)

```shell
bash infer.sh -d esconv -s sampling
```

#### Evaluation

the script `eval.sh` has one parameter, namely `d`.

- `d`: fine-tuned **d**ataset (`esconv`)

```shell
bash eval.sh -d esconv
```

