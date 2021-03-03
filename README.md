# DeepSeqSLAM - The Deep Learning Framework for High-Performance Long-Term Place Recognition #

This repository contains the official PyTorch implementation of the papers:

**[1] Sequential Place Learning: Heuristic-Free High-Performance Long-Term Place Recognition**
Marvin Chancán, Michael Milford. [\[ArXiv\]](https://arxiv.org/abs/2103.00000)  [\[Website\]](https://mchancan.github.io/spl) 

**[2] DeepSeqSLAM: A Trainable CNN+RNN for Joint Global Description and Sequence-based Place Recognition**.
Marvin Chancán, Michael Milford. [NeurIPS 2020](https://neurips.cc/Conferences/2020/) Workshop on Machine Learning for Autonomous Driving ([ML4AD](https://ml4ad.github.io)).
 [\[ArXiv\]](https://arxiv.org/abs/2011.08518) [\[Website\]](https://mchancan.github.io/deepseqslam) 

Both papers introduce DeepSeqSLAM, a CNN+LSTM `baseline` for state-of-the-art place recognition.
DeepSeqSLAM leverages `visual and positional` time-series data for joint global description and sequential place inference
in the context of simultaneous localization and mapping (SLAM) and autonomous driving research. Contrary to classical two-stage pipelines, *e.g.*,
*match-then-temporally-filter*, this codebase is orders of magnitud faster, scalable and learns from a single traversal of a route,
while accurately generalizing to multiple traversals of the same route under very different environmental conditions.

![baseline](https://user-images.githubusercontent.com/25828032/109748988-87dd6600-7c25-11eb-82c4-b8c3d298601a.png)


## BibTex Citation

If you find any of the tools provided here useful for your research or report our results in a publication,
please consider citing both [Sequential Place Learning](https://arxiv.org/abs/2103.00000) and [DeepSeqSLAM](https://arxiv.org/abs/2011.08518) papers:

```text
@article{chancan2021spl,
	author = {Marvin {Chanc\'an} and Michael {Milford}},
	title = {Sequential Place Learning: Heuristic-Free High-Performance Long-Term Place Recognition},
	journal = {arXiv preprint arXiv:2103.00000},
	year = {2021}
}

@article{chancan2020deepseqslam,
	author = {Marvin {Chanc\'an} and Marvin {Milford}},
	title = {DeepSeqSLAM: A Trainable CNN+RNN for Joint Global Description and Sequence-based Place Recognition},
	journal = {arXiv preprint arXiv:2011.08518},
	year = {2020}
}
```


## Getting Started

You just need Python v3.6+ with standard scientific packages, PyTorch v1.1+, and TorchVision v0.3.0+.

`git clone https://github.com/mchancan/deepseqslam`


## Training Data

The challenging [Gardens Point](https://zenodo.org/record/4561862) Walking dataset consists of three folders with 200 images each. The image name indicates correspondence in location between each of the three route traversals. Download the [dataset](https://zenodo.org/record/4561862), unzip, and place the `day_left`, `day_right`, and `night_right` image folders in the `datasets/GardensPointWalking` directory of DeepSeqSLAM.

## Single Node Training (CPU, GPU or Multi-GPUs)

In this release, we provide an implementation of DeepSeqSLAM for evaluation on the Gardens Point dataset with challenging day-night changing conditions. We also provide normalized (synthetic) positional data for end-to-end training and deployment.

### Run the demo on the Gardens Point dataset

```sh
sh demo_deepseqslam.sh
```

You can run this demo using one of these pre-trained models: `alexnet`, `resnet18`, `vgg16`, `squeezenet1_0`, `densenet161`, or easily configure the `run.py` script for training with any other [PyTorch's model](https://pytorch.org/docs/stable/torchvision/models.html) from `torchvision`.


### Commands example

```bash
SEQ_LENGHT=10

BATCH_SIZE=16

EPOCHS=100

NGPUS=1

SEQ1='day_left'

SEQ2='day_right'

SEQ3='night_right'

CNN='resnet18'

MODEL_NAME="gp_${CNN}_lstm"

python run.py train \
    --model_name $MODEL_NAME \
    --ngpus $NGPUS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LENGHT \
    --epochs $EPOCHS \
    --val_set $SEQ2 \
    --cnn_arch $CNN

for i in $SEQ1 $SEQ2 $SEQ3
do
    python run.py val \
    --model_name $MODEL_NAME \
    --ngpus $NGPUS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LENGHT \
    --val_set $i \
    --cnn_arch $CNN
done
```

### Help

```bash
usage: run.py [-h] [--data_path DATA_PATH] [-o OUTPUT_PATH]
              [--model_name MODEL_NAME] [-a ARCH] [--pretrained PRETRAINED]
              [--val_set VAL_SET] [--ngpus NGPUS] [-j WORKERS]
              [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
              [--load LOAD] [--nimgs NIMGS] [--seq_len SEQ_LEN]
              [--nclasses NCLASSES] [--img_size IMG_SIZE]

Gardens Point Training

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to dataset folder that contains preprocessed
                        train and val *npy image files
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path for storing model checkpoints
  --model_name MODEL_NAME
                        checkpoint model name (default:
                        deepseqslam_resnet18_lstm)
  -a ARCH, --cnn_arch ARCH
                        model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn (default: resnet18)
  --pretrained PRETRAINED
                        use pre-trained CNN model (default: True)
  --val_set VAL_SET     validation_set (default: day_right)
  --ngpus NGPUS         number of GPUs for training; 0 if you want to run on
                        CPU (default: 2)
  -j WORKERS, --workers WORKERS
                        number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 200)
  --batch_size BATCH_SIZE
                        mini-batch size: 2^n (default: 32)
  --lr LR, --learning_rate LR
                        initial learning rate (default: 1e-3)
  --load LOAD           restart training from last checkpoint
  --nimgs NIMGS         number of images (default: 200)
  --seq_len SEQ_LEN     sequence length: ds (default: 10)
  --nclasses NCLASSES   number of classes = nimgs - seq_len (default: 190)
  --img_size IMG_SIZE   image size (default: 224)
```

## Multiple Nodes

For training on multiple nodes, you should use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance. Please refer to [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet) for additional information on this.

## Acknowledgements

This code has been largely inspired by the following projects:

- [https://github.com/dicarlolab/CORnet](https://github.com/dicarlolab/CORnet)
- [https://github.com/dicarlolab/vonenet](https://github.com/dicarlolab/vonenet)
- [https://github.com/mchancan/flynet](https://github.com/mchancan/flynet)
- [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

## License (Updated as of 27/11/2020)

GNU GPL 3+
