# DL4NMT_Theano

Deep neural machine translation (NMT) model, implemented in Theano.

Dual learning will be included.

## Install in new node

```bash

git clone https://github.com/fyabc/DL4NMT_Theano.git
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
pip install .
cd ../DL4NMT_Theano
mkdir -p data/train data/test data/dev data/dic model/complete
mkdir -p log/complete translated/complete
# copy data from other nodes to here...
```
