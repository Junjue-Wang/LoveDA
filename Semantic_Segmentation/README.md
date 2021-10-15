<h2 align="center">LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation</h2>


<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Ailong Ma, Xiaoyan Lu, and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>




This is an initial benchmark for Land-cover Semantic Segmentation.
---------------------


## Getting Started


#### Requirements:
- pytorch >= 1.1.0
- python >=3.6

### Install Ever + Segmentation Models PyTorch
```bash
pip install --upgrade git+https://gitee.com/zhuozheng/ever_lite.git@v1.4.5
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```

### Evaluate Model 
Some examples:
```bash
bash ./scripts/eval_deeplabv3p.sh
bash ./scripts/eval_hrnetw32.sh
```

### Train Model
```bash 
bash ./scripts/train_deeplabv3p.sh
```

### Predict Model
```bash 
bash ./scripts/predict.sh
```

Submit your test results on [LoveDA Semantic Segmentation Challenge](https://competitions.codalab.org/competitions/35865#) and you will get your Test score.

