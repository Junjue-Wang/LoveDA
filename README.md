<h2 align="center">LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation</h2>

<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Ailong Ma, Xiaoyan Lu, and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

[[`Paper`](https://www.researchgate.net/profile/Junjue-Wang/publication/355390292_LoveDA_A_Remote_Sensing_Land-Cover_Dataset_for_Domain_Adaptive_Semantic_Segmentation/links/617cd8bda767a03c14cecbc9/LoveDA-A-Remote-Sensing-Land-Cover-Dataset-for-Domain-Adaptive-Semantic-Segmentation.pdf?_sg%5B0%5D=Iw5FPui1-9iYrZN7aZO766hZA-LmublHlq8bp0694vUeIGDIzp5SGTfYN-OWhurZOujSPU0RDZ5lW0i02HVUew.7x9qdrvJwRmAnsqEyh5-xSFdh0M9AaTpdXcZCfHyhVl5GNLR5nlDIx8ctTXFy1HE1yNexX4ytzYqJWkAGJVTvg.Rrg3rXhcp9mMlLTU3n9Jf-h0Kt8VzHAd0AmhG2yPQxI-yRK6J0wAulUZ65dih6BQ9CbrQm0_23_nULO_BXwaJg&_sg%5B1%5D=KLu7pn0g50f8FwKE9x5iOuDPYb8VaOpX4k_ieq8eWJVVeJyXbZJO-O4pCL687QRxYbBnWdo7fJj8FZEOc3t3lgVVyDz0CFS-ff7LToXj4R9W.7x9qdrvJwRmAnsqEyh5-xSFdh0M9AaTpdXcZCfHyhVl5GNLR5nlDIx8ctTXFy1HE1yNexX4ytzYqJWkAGJVTvg.Rrg3rXhcp9mMlLTU3n9Jf-h0Kt8VzHAd0AmhG2yPQxI-yRK6J0wAulUZ65dih6BQ9CbrQm0_23_nULO_BXwaJg&_iepl=)],
[[`Video`](https://slideslive.com/38969542)],
[[`Dataset`](https://doi.org/10.5281/zenodo.5706578)],
[[`BibTeX`](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Bibtex-round2.bib)],
[[`Leaderboard-SEG`](https://codalab.lisn.upsaclay.fr/competitions/421)],
[[`Leaderboard-UDA`](https://codalab.lisn.upsaclay.fr/competitions/424)]

<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/LoveDA/LoveDA.jpg?raw=true">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/LoveDA/statics_diff.png?raw=true">
</div>

## News
- 2024/05/12, The new version of LoveDA dataset has been released at [<b>EarthVQA</b>](https://github.com/Junjue-Wang/EarthVQA) dataset.

- 2021/12/13, Pre-trained urls for HRNet have been updated.

- 2021/12/10, LoveDA has been included in
[<b>Torchgeo</b>](https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/loveda.py).

- 2021/11/30, The contests have been moved to new server:
[<b>LoveDA Semantic Segmentation Challenge</b>](https://codalab.lisn.upsaclay.fr/competitions/421), [<b>LoveDA Unsupervised Domain Adaptation Challenge</b>](https://codalab.lisn.upsaclay.fr/competitions/424).

- 2021/11/11, LoveDA has been included in [MMsegmentation](https://github.com/open-mmlab/mmsegmentation).
🔥🔥 The Semantic Segmentation task can be prepared follow [dataset_prepare.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda).🔥🔥 




## Highlights
1. 5987 high spatial resolution (0.3 m) remote sensing images from Nanjing, Changzhou, and Wuhan
2. Focus on different geographical environments between Urban and Rural
3. Advance both semantic segmentation and domain adaptation tasks
4. Three considerable challenges:
    * Multi-scale objects
    * Complex background samples
    * Inconsistent class distributions

## Citation
If you use LoveDA in your research, please cite our NeurIPS2021 paper.
```text
    @inproceedings{NEURIPS DATASETS AND BENCHMARKS2021_4e732ced,
         author = {Wang, Junjue and Zheng, Zhuo and Ma, Ailong and Lu, Xiaoyan and Zhong, Yanfei},
         booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
         editor = {J. Vanschoren and S. Yeung},
         pages = {},
         publisher = {Curran Associates, Inc.},
         title = {LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
         url = {https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf},
         volume = {1},
         year = {2021}
    }
    @dataset{junjue_wang_2021_5706578,
        author={Junjue Wang and Zhuo Zheng and Ailong Ma and Xiaoyan Lu and Yanfei Zhong},
        title={Love{DA}: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
        month=oct,
        year=2021,
        publisher={Zenodo},
        doi={10.5281/zenodo.5706578},
        url={https://doi.org/10.5281/zenodo.5706578}
    }
```


## Dataset and Contest
The LoveDA dataset is released at [<b>Zenodo</b>](https://doi.org/10.5281/zenodo.5706578)
and [<b>Baidu Drive</b>](https://pan.baidu.com/s/1YrU1Y4Y0dS0f_OOHXpzspQ) Code: 27vc



You can develop your models on Train and Validation sets.

Category labels: background – 1, building – 2, road – 3,
                 water – 4, barren – 5,forest – 6, agriculture – 7. And the no-data regions were assigned 0
                 which should be ignored. The provided data loader will help you construct your pipeline.  
                 

Submit your test results on [<b>LoveDA Semantic Segmentation Challenge</b>](https://codalab.lisn.upsaclay.fr/competitions/421), [<b>LoveDA Unsupervised Domain Adaptation Challenge</b>](https://codalab.lisn.upsaclay.fr/competitions/424).
You will get your Test scores smoothly.

Feel free to design your own models, and we are looking forward to your exciting results!


## License
The owners of the data and of the copyright on the data are [RSIDEA](http://rsidea.whu.edu.cn/), Wuhan University.
Use of the Google Earth images must respect the ["Google Earth" terms of use](https://about.google/brand-resource-center/products-and-services/geo-guidelines/).
All images and their associated annotations in LoveDA can be used for academic purposes only,
<font color="red"><b> but any commercial use is prohibited.</b></font>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">
<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
