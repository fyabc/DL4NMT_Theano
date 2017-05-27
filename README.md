# DL4NMT_Theano

Deep neural machine translation (NMT) model, implemented in Theano.

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

**NOTE**：在node（包括GCR）上跑job之前，请确保code是最新的，在Project根目录下运行`git pull`。

## Train

Run `train_nmt.py`.

See `train_nmt.py -h` for help.

用于交GCR job的脚本见OneNote:/AMD Notebook/Yang Fan/Jobs (Complete, with Fei & Yingce) 里面的几个脚本。

**NOTE**：由于shuffle data per epoch的存在，当一个job使用的dataset没有shuffle版本（下标为0,1,2,...）时，会立即创建一个。
因此**不要**同时交两个dataset没有shuffle版本的job，防止冲突。等一个把下标为0的shuffle版本创建出来之后再交另外的。

## Options

所有options见`config.py`。

可配置的options见`train_nmt.py`。

## Dataset

dataset由`--dataset` option控制，所有dataset见`constants.py`。

dataset由training data，small training data，validation data，dictionary组成。

目前有些truecase版本的dataset还不全，需要补充。

### 对dataset进行Truecase转换

运行`make_truecase.bat`脚本，要求原始dataset分别在各自目录下（`data/train, data/test, data/dev, data/dic`）。

该脚本具体用法见其中Usage行，最后一个可选项为single_dict，若设置，则source和target使用同一个dict，名字为single_dict。


## Test

Test脚本以"test_"开头。


以_single结尾的脚本用于linux server，其余用于windows server。

BPE test脚本在Windows上需要用bash运行，gdw135和gdw144均安装了bash。

脚本用法可见脚本里面的Usage行。


Linux server脚本默认model path为/datadrive/v-yanfa/model/complete。

Windows server脚本默认model path为//gcr/Scratch/RR1/v-yanfa/SelectiveTrain/model/complete。


zh-en的test方法需要问lijun。

truecase的test脚本暂时还没有，根据[这里](http://www.statmt.org/moses/?n=Moses.SupportTools#ntoc11)，
应该在运行`perl multi-bleu.perl`之前运行`perl detruecase.perl < translated_file > output`。
