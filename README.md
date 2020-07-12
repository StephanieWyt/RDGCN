# RDGCN

Source code and datasets for IJCAI 2019 paper: ***[Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs](https://arxiv.org/pdf/1908.08210.pdf)***.

Initial datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

## Dependencies

* Python>=3.5
* Tensorflow>=1.8.0
* Scipy>=1.1.0
* Numpy

> Due to the limited graphics memory of GPU, we ran our codes using CPUs (40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz).

## Datasets

Please first download the datasets [here](https://drive.google.com/drive/folders/13u-4r4aJbjhUPRbDXrVFA3QfQS0y_8Ye?usp=sharing) and extract them into `data/` directory.

There are three cross-lingual datasets in this folder:
- fr-en
- ja-en
- zh-en

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;

## Running

* Modify language or some other settings in *include/Config.py*
* cd to the directory of *main.py*
* run *main.py*

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to wyting@pku.edu.cn.

## Citation

If you use this model or code, please cite it as follows:

*Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui Yan, Dongyan Zhao. Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pages 5278-5284, 2019.*

```
@inproceedings{ijcai2019-733,
  title={Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs},
  author={Wu, Yuting and Liu, Xiao and Feng, Yansong and Wang, Zheng and Yan, Rui and Zhao, Dongyan},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},            
  pages={5278--5284},
  year={2019},
}
```
