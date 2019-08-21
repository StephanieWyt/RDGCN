# RDGCN

Source code and datasets for IJCAI 2019 paper: ***Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs***.

Initial datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

## Environment

* Python=3.5.2
* Tensorflow=1.8.0
* Scipy=1.1.0
* Numpy=1.15.4

## Dataset

You could download our demo dataset *[here](http://59.108.48.35/data.tar.gz)* into `data/` directory.

## Running

* Modify language or some other settings in *include/Config.py*
* cd to the directory of *main.py*
* run *main.py*

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to wyting@pku.edu.cn.

## Citation

If you use this model or code, please cite it as follows:

*Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui Yan, Dongyan Zhao. Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pages 5278-5284, 2019.*

@inproceedings{ijcai2019-733,
  title={Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs},
  author={Wu, Yuting and Liu, Xiao and Feng, Yansong and Wang, Zheng and Yan, Rui and Zhao, Dongyan},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},            
  pages={5278--5284},
  year={2019},
}
