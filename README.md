# ScaleFL: Resource-Adaptive Federated Learning with Heterogeneous Clients

Code for the following paper:

Fatih Ilhan, Gong Su and Ling Liu, "ScaleFL: Resource-Adaptive Federated Learning with Heterogeneous Clients," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, Canada, Jun. 18-22, 2023.

## Introduction

Federated learning (FL) is an attractive distributed learning paradigm supporting real-time continuous learning and client privacy by default. In most FL approaches, all edge clients are assumed to have sufficient computation capabilities to participate in the learning of a deep neural network (DNN) model. However, in real-life applications, some clients may have severely limited resources and can only train a much smaller local model. This paper presents ScaleFL, a novel FL approach with two distinctive mechanisms to handle resource heterogeneity and provide an equitable FL framework for all clients. First, ScaleFL adaptively scales down the DNN model along width and depth dimensions by leveraging early exits to find the best-fit models for resource-aware local training on distributed clients. In this way, ScaleFL provides an efficient balance of preserving basic and complex features in local model splits with various sizes for joint training while enabling fast inference for model deployment. Second, ScaleFL utilizes self-distillation among exit predictions during training to improve aggregation through knowledge transfer among subnetworks. We conduct extensive experiments on benchmark CV (CIFAR-10/100, ImageNet) and NLP datasets (SST-2, AgNews). We demonstrate that ScaleFL outperforms existing representative heterogeneous FL approaches in terms of global/local model performance and provides inference efficiency, with up to 2x latency and 4x model size reduction with negligible performance drop below 2%.

## Requirements
* Python 3.7
* PyTorch 1.12
* HuggingFace transformer 4.21
* HuggingFace datasets 2.4

## Usage

### Train ResNet110 on CIFAR10:
python main.py --data-root {data-root} --data cifar10 --arch resnet110_4 --use-valid 

### Train MSDNet24 on CIFAR100:
python main.py --data-root {data-root} --data cifar100 --arch msdnet24_4 --use-valid --ee_locs 15 18 21 --vertical_scale_ratios 0.65 0.7 0.85 1

### Train EffNetB4 on ImageNet:
python main.py --data-root {data-root} --data imagenet --arch effnetb4_4 --use-valid --num_rounds 90 --num_clients 50 --sample_rate 0.2 --vertical_scale_ratios 0.65 0.65 0.82 1

### Train BERT on AgNews:
python main.py --data-root {data-root} --data ag_news --arch bert_4 --use-valid --ee_locs 4 6 9 --KD_gamma 0.05 --num_rounds 100 --num_clients 50 --sample_rate 0.2 --vertical_scale_ratios 0.4 0.55 0.75 1

### Parameters

All training/inference/model parameters are controlled from ``config.py``.