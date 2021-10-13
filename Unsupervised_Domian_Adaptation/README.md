<h2 align="center">LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation</h2>



<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Ailong Ma, Xiaoyan Lu, and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>



This is an initial benchmark for Unsupervised Domain Adaptation.
---------------------


## Getting Started
### Install Ever_lite

```bash
pip install --upgrade git+https://gitee.com/zhuozheng/ever_lite.git@v1.4.5
```

#### Requirements:
- pytorch >= 1.7.0
- python >=3.6
- pandas >= 1.1.5
### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```

### Evaluate CBST Model 
From Rural to Urban
```bash
bash ./scripts/eval_cbst.sh
```

### Train CBST Model
From Rural to Urban
```bash 
bash ./scripts/train_cbst.sh
```


