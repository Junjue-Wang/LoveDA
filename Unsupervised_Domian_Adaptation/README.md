<h2 align="center">LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation</h2>



<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Ailong Ma, Xiaoyan Lu, and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>



This is an initial benchmark for Unsupervised Domain Adaptation.
---------------------


## Getting Started

#### Requirements:
- pytorch >= 1.7.0
- python >=3.6
- pandas >= 1.1.5
### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```


### Evaluate CBST Model on the predict set
#### 1. Download the pre-trained [<b>weights</b>](https://drive.google.com/drive/folders/1xFn1d8a4Hv4il52hLCzjEy_TY31RdRtg?usp=sharing)
#### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./CBST_2Urban.pth ./log/CBST_2Urban.pth
```

#### 3. Evaluate on Urban test set
```bash 
bash ./scripts/predict_cbst.sh
```
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.

### Train CBST Model
From Rural to Urban
```bash 
bash ./scripts/train_cbst.sh
```
Eval CBST Model on Urban val set
```bash
bash ./scripts/eval_cbst.sh
```







