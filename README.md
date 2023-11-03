
# Trustworthy Multimodal Regression with Mixture of Normal-inverse Gamma Distributions

This repository is the code for paper "Trustworthy Multimodal Regression with Mixture of Normal-inverse Gamma Distributions".

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```


## Download

For convenience, we pack the dataset and it is available on 

```
link：https://pan.baidu.com/s/1xgt3o3HHPlWyx501pQ071A
code：8s73
```
.pkl and .pt indicate original data and torch tensors respectively. 

## Inference

```
python3 main.py --model MOE --dataset mosei_senti
```




## Reference


```
@article{ma2021trustworthy,
  title={Trustworthy multimodal regression with mixture of normal-inverse gamma distributions},
  author={Ma, Huan and Han, Zongbo and Zhang, Changqing and Fu, Huazhu and Zhou, Joey Tianyi and Hu, Qinghua},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={6881--6893},
  year={2021}
}
```
