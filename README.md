<h2 align="center">LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation</h2>

[`Paper`]

<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/LoveDA/LoveDA.jpg?raw=true">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/LoveDA/statics_diff.png?raw=true">
</div>

## News
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
If you use LoveDA in your research, please cite our coming NeurIPS2021 paper.
```text
    @inproceedings{wang2021loveda,
        title={Love{DA}: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
        author={Junjue Wang and Zhuo Zheng and Ailong Ma and Xiaoyan Lu and Yanfei Zhong},
        booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
        editor = {J. Vanschoren and S. Yeung},
        year={2021},
        volume = {1},
        pages = {},
        url={https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf}
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
The LoveDA dataset is released at [<b>Zenodo</b>](https://doi.org/10.5281/zenodo.5706578), 
[<b>Google Drive</b>](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing)
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