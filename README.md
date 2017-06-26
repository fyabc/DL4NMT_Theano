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


## TODO

1. 师弟，现在有一些任务需要你来做一下。你先读一下附件里的这篇paper，我们想要做一个类似这个paper里面的事情，
就是visualize一下 deep NMT model里面不同layers的gates/activation的值。你读完之后我们再具体讨论下应该怎么做。
2. 另外你改一下现在的code，把fine tune阶段改成这样的feature:
    先halve lr，再halve grad clip threshold，再halve lr， 再clip threshold，再lr ......

    什么时候决定halve取决于dev集上的bleu score，即类似于原始code里面的early stop，如果patience个check point
    （就是validation points），dev 集上的bleu仍然没涨，那么就进行halve lr/clip value。所以你需要加一个subprocess，就是get dev集上的bleu score。
所以也就是说以后我们不用--lr_discount 这个参数，而是改成一个类似lr_discount_patience这样的参数，默认可以是10，
然后在get dev bleu的时候，为了快速计算可以用beam_size=2。


## NOTES

Scripts in `scripts` must be call at root directory of the project (the directory of this README).
