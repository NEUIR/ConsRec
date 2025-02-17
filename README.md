# ConsRec

Source code for our paper: Denoising Sequential Recommendation through User-Consistent Preference Modeling.

If you find this work useful, please cite our paper and give us a shining star 🌟

```bibtex
```

## Overview

ConsRec constructs a user-interacted item graph, learns item similarities from their text representations, and then extracts the maximum connected subgraph from the user-interacted item graph for denoising items. Notably, ConsRec shows the generalization ability by broadening its advantages to both item ID-based and text-based recommendation models.

![The Model Architecture of ConsRec](figs/model.png)

## Requirement

1.Install the following packages using Pip or Conda under this environment.

```
python >= 3.8
torch == 1.12.1
recbole == 1.2.0
datasets == 3.1.0
transformers == 4.22.2
sentencepiece == 0.2.0
faiss-cpu == 1.8.0.post1
```

2.Install openmatch. More details can be found at [https://github.com/OpenMatch/OpenMatch](https://github.com/OpenMatch/OpenMatch)

```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e.
```

3.Prepare the pretrained T5 weights.

```
git lfs install
git clone https://huggingface.co/google-t5/t5-base
```

## Reproduce ConsRec

### 1. 数据集处理

### 2. 构建mfilter训练数据

### 3. 预训练mfilter

### 4. 使用mfilter生成嵌入表示

### 5. 计算最大连通子图

### 6. 使用recbole处理数据

### 7. mrec训练

### 8. mrec评估

### 9. mrec测试

## Contact

If you have questions, suggestions, and bug reports, please email:

```
xinhaidong@stumail.neu.edu.cn
```