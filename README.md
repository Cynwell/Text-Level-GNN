# Text-Level-GNN
An implementation to the paper: Text Level Graph Neural Network for Text Classification (https://arxiv.org/pdf/1910.02356.pdf)

## Features:
- Dynamic edge weights instead of static edge weights
- All documents are from a big graph instead of every documents having its own structure
- Public edge sharing (achieved by computing edge statistics during dataset construction and masking during training, a novel mechanism roughly described by the paper yet without much further information)
- Flexible argument controls and early stopping features
- Detailed explanations about intermediate operations
- The number of parameters in this model is close to the amount of parameters mentioned in the paper

## File structure:
```
+---embeddings\
|             +---glove.6B.50d.txt
|             +---glove.6B.100d.txt
|             +---glove.6B.200d.txt
|             +---glove.6B.300d.txt
+---train.py
+---r8-test-all-terms.csv
+---r8-train-all-terms.csv
```

## Environment:
- Python 3.7.4
- PyTorch 1.5.1 + CUDA 10.1
- Pandas 1.0.5
- Numpy 1.19.0

Successful run on RTX 2070, RTX 2080 Ti and RTX 3090. However, the memory consumption is quite large that it requires smaller batch size / shorter MAX_LENGTH / smaller embedding_size on RTX 2070.

## Usage:
- Linux:
  - OMP_NUM_THREADS=1 python train.py --cuda=0 --embedding_size=300 --p=3 --min_freq=2 --max_length=70 --dropout=0 --epoch=300
- Windows:
  - python train.py --cuda=0 --embedding_size=300 --p=3 --min_freq=2 --max_length=70 --dropout=0 --epoch=300

## Result:
I only tested the model on r8 dataset and is unable to achieve the figure as described in the paper despite having tried some hyperparameter tunings. The closest run that I could get is:
Train Accuracy|Validation Accuracy|Test Accuracy
:---:|:---:|:---:
99.91%|95.7%|96.2%

with `embedding_size=300`, `p=3` and `70<=max_length<=150` and `dropout=0`.
As the experiment settings described in the paper is not clearly stated, I assumed they used a learning rate decay mechanism too. I also added a warming up mechanism to pretrain the model. But actually the model converged quite fast and does not even need to use warming up technique.
