# Overview
Original paper -> [N2D: (Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding](https://arxiv.org/abs/1908.05968)

# Requirement
```
pip install umap-learn
```

# MNIST Result
- training data
![](assets/train.png)

- test data
![](assets/test.png)

# Usage 
```
python train.py -bs 256 -k 10 -epoch 30
```
