# ConsRec: Denoising Sequential Recommendation through User-Consistent Preference Modeling

This repository contains the source code for the paper: [Denoising Sequential Recommendation through User-Consistent Preference Modeling]().

## Overview

ConsRec addresses the challenge of noisy data in sequential recommendation by constructing a user-interacted item graph. It leverages item similarities derived from their text representations to extract the maximum connected subgraph, effectively denoising the items a user has interacted with. ConsRec demonstrates strong generalization capabilities by enhancing both item ID-based and text-based recommendation models.

![](figs/model.png)

## Key Features

- **User-Consistent Preference Modeling:** Captures consistent user preferences by leveraging item similarities.

- **Denoising via Graph Extraction:**  Identifies and removes noisy items through maximum connected subgraph extraction.

- **Generalizability:** Improves the performance of various recommendation models.

## Requirements

### 1. Python Environment:

Install the following packages using Pip or Conda under this environment.

```
python >= 3.8
torch == 1.12.1
recbole == 1.2.0
datasets == 3.1.0
transformers == 4.22.2
sentencepiece == 0.2.0
faiss-cpu == 1.8.0.post1
scikit-learn >= 1.1.2
numpy >= 1.17.2
pandas >= 1.0.0
tqdm
jsonlines
networkx
```

### 2. Install Openmatch.

Refer to [https://github.com/OpenMatch/OpenMatch](https://github.com/OpenMatch/OpenMatch) for detailed instructions.

```bash
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e .
```

### 3. Pretrained T5 weights.

Download pretrained T5 weights from Hugging Face.

```bash
git lfs install
git clone https://huggingface.co/google-t5/t5-base
```

*Note:* Ensure that `git lfs` is properly installed. You may need to run `git lfs install` before cloning the T5 weights.

## Reproducion Guide

This section provides a step-by-step guide to reproduce the ConsRec results.

### 1. Dataset Preprocessing

We utilize the Amazon Product 2014 and Yelp 2020 datasets. Download the original data from:

- [Amazon Product 2014](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
- [Yelp 2020](https://business.yelp.com/data/resources/open-dataset/)

The following example uses the Amazon Beauty dataset.

#### 1.1. Download and Prepare Amazon Beauty Dataset:

```bash
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
```

#### 1.2. Unzip the Metadata File:

```bash
gzip -d meta_Beauty.json.gz
```

#### 1.3. Organize Files:

```bash
mkdir data
mv ratings_Beauty.csv data/
mv meta_Beauty.json data/
```

#### 1.4. Process Raw Data for Recbole:

```bash
mkdir dataset
bash scripts/process_origin.sh
```

#### 1.5. Extract and Process Required Data:

```bash
bash scripts/process_beauty.sh
```

### 2. Data Preprocessing for Training $\text{M}_{Filter}$

Before proceeding, process all four original datasets as described above to obtain the atomic files. Then, construct the mixed pretraining data for \text{M}_{Filter} according to your desired proportions.

#### 2.1. Construct Training and Test Data using Recbole:

```bash
bash scripts/gen_dataset.sh
```

#### 2.2. Generate Item Representations using $\text{M}_{Rec}$:

```bash
bash scripts/gen_pretrain_items.sh
```

#### 2.3. Sample Training Data for $\text{M}_{Filter}$:

For $\text{M}_{Filter}$ training data construction, we sampled the four datasets with balance. For each dataset, we selected the number of items corresponding to the dataset with the largest number of training samples and then randomly supplemented the datasets with insufficient training data:

```bash
python src/sample_train.py
```

#### 2.4. Sample Validation Data:

Similarly, we selected the number of training samples from the dataset with the fewest training items in each case to serve as the validation set:

```bash
python src/sample_valid.py
```

#### 2.5. Construct Pretraining Data for Sampled Items:

```bash
bash scripts/build_pretrain.sh
```

#### 2.6. Merge Training and Validation Data:

```bash
python src/merge_json.py
```

### 3. Pretraining for $\text{M}_{Filter}$

Pretrain the T5 model using next item prediction (NIP) and mask item prediction (MIP) tasks.

```bash
bash scripts/train_mfilter.sh
```

Adjust training parameters based on your GPU device. Select the checkpoint with the lowest evaluation loss as the final $\text{M}_{Filter}$ .

### 4. Generate Embedding Representations using \text{M}_{Filter}

Save the item embedding representations to avoid redundant calculations.

```bash
mkdir embedding
bash scripts/gen_gembeddings.sh
```

### 5. Denoise Dataset by Calculating the Maximum Connected Subgraph

Embed the nodes into an undirected graph and use BFS to calculate the maximum connected subgraph.

```bash
bash scripts/build_graph.sh
```

Copy the original item information file to the denoised data folder.

```bash
cp dataset/beauty/beauty.item dataset/beauty_filtered/
mv dataset/beauty_filtered/beauty.item dataset/beauty_filtered/beauty_filtered.item
```

### 6. Build Standardized Training Data for $\text{M}_{Rec}$ using Recbole

```bash
bash scripts/gen_dataset.sh
bash scripts/gen_train_items.sh
bash scripts/build_train.sh
```

### 7. Training $\text{M}_{Rec}$

```bash
bash scripts/train_mrec.sh
```

### 8. Evaluate $\text{M}_{Rec}$

```bash
bash scripts/eval_mrec.sh
```

### 9. Test $\text{M}_{Rec}$

```bash
bash scripts/test_mrec.sh
```

## Experimental Result

![](figs/result.png)

## Acknowledgement

- [OpenMatch](https://github.com/OpenMatch/OpenMatch): We utilize OpenMatch to reproduce the $\text{M}_{Rec}$ module.
- [Recbole](https://github.com/RUCAIBox/RecBole): We leverage RecBole for dataset processing and baseline reproduction.

## Citation

If you find this work useful, please cite our paper and give us a shining star 🌟

```bibtex
@inproceedings{xin2025consrec,
  title={Denoising Sequential Recommendation through User-Consistent Preference Modeling},
  author={},
  journal={},
  year={2025}
}
```

## Contact

For questions, suggestions, or bug reports, please contact:

```
xinhaidong@stumail.neu.edu.cn
```