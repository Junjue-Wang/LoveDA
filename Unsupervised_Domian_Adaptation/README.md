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


### Evaluate CBST Model on the eval set
#### 1. Download the pre-trained [<b>weights</b>](https://github.com/Junjue-Wang/LoveDA/releases/tag/v0.2.0-alpha)
#### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./CBST_2Urban.pth ./log/CBST_2Urban.pth
```

#### 3. Evaluate on Urban eval set
```bash
bash ./scripts/eval_cbst.sh
```

### Train CBST Model
From Rural to Urban
```bash 
bash ./scripts/train_cbst.sh
```
### Predict on test set
Please see the [guidelines](https://github.com/Junjue-Wang/LoveDA/blob/17d3725fc0f725f42b69a6450a6cd724c098b634/Semantic_Segmentation/predict.py) in our provided Semantic Segmentation.

Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://competitions.codalab.org/competitions/35865#) and you will get your Test score.



