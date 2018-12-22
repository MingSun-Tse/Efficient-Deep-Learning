# EfficientDNNs
A collection of recent methods on DNN compression and acceleration. There are mainly 5 kinds of methods for efficient DNNs:
- neural architecture re-designing or searching
- pruning (including structured and unstructured)
- quantization
- matrix decomposition
- knowledge distillation

> PS. In the list below, o for oral, w for workshop, s for spotlight, b for best paper.

### Papers
- 2015-CVPR-Fully convolutional networks for semantic segmentation
- 2015-CVPR-Learning to generate chairs with convolutional neural networks
- 2015-CVPR-Single image super-resolution from transformed selfexemplars
- 2015-ICCV-Deeply improved sparse coding for image super-resolution
- 2015-PAMI-Image super-resolution using deep convolutional networks [[paper](https://arxiv.org/pdf/1501.00092.pdf)]
- 2016-ICLR-All you need is a good init
- 2016-CVPR-Accurate image super-resolution using very deep convolutional networks
- 2016-ECCV-Accelerating the Super-Resolution Convolutional Neural Network [[paper](https://arxiv.org/abs/1608.00367)][[code](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)]
- 2016-EMNLP-Sequence-Level Knowledge Distillation [[paper](https://arxiv.org/abs/1606.07947)]
- 2017-CVPR-All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation
- 2018-AAAI-Auto-balanced Filter Pruning for Efficient Convolutional Neural Networks [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16450/16263)]
- 2018-AAAI-Deep Neural Network Compression with Single and Multiple Level Quantization [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16479/16742)]

- 2018-ICLRo-Training and Inference with Integers in Deep Neural Networks [[paper](https://openreview.net/forum?id=HJGXzmspb)]
- 2018-ICLR-Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers [[paper](https://openreview.net/forum?id=HJ94fqApW)]
- 2018-ICLR-N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning [[paper](https://openreview.net/forum?id=B1hcZZ-AW)]
- 2018-ICLR-Model compression via distillation and quantization [[paper](https://openreview.net/forum?id=S1XolQbRW)]
- 2018-ICLR-Towards Image Understanding from Deep Compression Without Decoding [[paper](https://openreview.net/forum?id=HkXWCMbRW)]
- 2018-ICLR-Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training [[paper](https://openreview.net/forum?id=SkhQHMW0W)]
- 2018-ICLR-Mixed Precision Training of Convolutional Neural Networks using Integer Operations [[paper](https://openreview.net/forum?id=H135uzZ0-)]
- 2018-ICLR-Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy [[paper](https://openreview.net/forum?id=B1ae1lZRb)]
- 2018-ICLR-Loss-aware Weight Quantization of Deep Networks [[paper](https://openreview.net/forum?id=BkrSv0lA-)]
- 2018-ICLR-Alternating Multi-bit Quantization for Recurrent Neural Networks [[paper](https://openreview.net/forum?id=S19dR9x0b)]
- 2018-ICLR-Adaptive Quantization of Neural Networks [[paper](https://openreview.net/forum?id=SyOK1Sg0W)]
- 2018-ICLR-Variational Network Quantization [[paper](https://openreview.net/forum?id=ry-TW-WAb)]
- 2018-ICLRw-To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression [[paper](https://openreview.net/forum?id=Sy1iIDkPM)]
- 2018-ICLRw-Systematic Weight Pruning of DNNs using Alternating Direction Method of Multipliers [[paper](https://openreview.net/forum?id=B1_u3cRUG)]
- 2018-ICLRw-Weightless: Lossy weight encoding for deep neural network compression [[paper](https://openreview.net/forum?id=rJpXxgaIG)]
- 2018-ICLRw-Variance-based Gradient Compression for Efficient Distributed Deep Learning [[paper](https://openreview.net/forum?id=Sy6hd7kvM)]
- 2018-ICLRw-Stacked Filters Stationary Flow For Hardware-Oriented Acceleration Of Deep Convolutional Neural Networks [[paper](https://openreview.net/forum?id=HkeAoQQHM)]
- 2018-ICLRw-Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks [[paper](https://openreview.net/forum?id=BJbtuRRLM)]
- 2018-ICLRw-Accelerating Neural Architecture Search using Performance Prediction [[paper](https://openreview.net/forum?id=HJqk3N1vG)]
- 2018-ICLRw-Nonlinear Acceleration of CNNs [[paper](https://openreview.net/forum?id=HkNpF_kDM)]
- 2018-CVPR-Context-Aware Deep Feature Compression for High-Speed Visual Tracking [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Context-Aware_Deep_Feature_CVPR_2018_paper.pdf)]
- 2018-ICMLw-Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures [[paper](https://openreview.net/forum?id=SyPicjbWQ)]
- 2018-IJCAI-Efficient DNN Neuron Pruning by Minimizing Layer-wise Nonlinear Reconstruction Error
- 2018-IJCAI-Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks
- 2018-IJCAI-Where to Prune: Using LSTM to Guide End-to-end Pruning
- 2018-IJCAI-Accelerating Convolutional Networks via Global & Dynamic Filter Pruning
- 2018-IJCAI-Optimization based Layer-wise Magnitude-based Pruning for DNN Compression
- 2018-IJCAI-Progressive Blockwise Knowledge Distillation for Neural Network Acceleration
- 2018-IJCAI-Complementary Binary Quantization for Joint Multiple Indexing
- 2018-ECCV-A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf)]
- 2018-ECCV-Coreset-Based Neural Network Compression [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Abhimanyu_Dubey_Coreset-Based_Convolutional_Neural_ECCV_2018_paper.pdf)]
- 2018-ECCV-Data-Driven Sparse Structure Selection for Deep Neural Networks [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zehao_Huang_Data-Driven_Sparse_Structure_ECCV_2018_paper.pdf)][[code](https://github.com/TuSimple/sparse-structure-selection)]
- 2018-BMVCo-Structured Probabilistic Pruning for Convolutional Neural Network Acceleration [[paper](http://bmvc2018.org/contents/papers/0870.pdf)]
- 2018-BMVC-Efficient Progressive Neural Architecture Search [[paper](http://bmvc2018.org/contents/papers/0291.pdf)]
- 2018-NIPS-Discrimination-aware Channel Pruning for Deep Neural Networks
- 2018-NIPS-Frequency-Domain Dynamic Pruning for Convolutional Neural Networks
- 2018-NIPS-ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions
- 2018-NIPS-DropBlock: A regularization method for convolutional networks [[paper]](http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks)
- 2018.11-Second-order Optimization Method for Large Mini-batch: Training ResNet-50 on ImageNet in 35 Epochs [[paper](https://arxiv.org/abs/1811.12019)]
- 2018.11-Rethinking ImageNet Pre-training [[paper](https://arxiv.org/abs/1811.08883)]
- 2018.05-Compression of Deep Convolutional Neural Networks under Joint Sparsity Constraints [[paper](https://arxiv.org/abs/1805.08303)]

## News
- [VALSE 2018年度进展报告 | 深度神经网络加速与压缩 (in Chinese)](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247485375&idx=1&sn=a066fe6f57e6d152719b8815af87e819&chksm=97d7e228a0a06b3e5be18face8716e1ad41d598c2676429a1d174659a384e5f0d5a9be2204ee#rd)
- [机器之心-腾讯AI Lab PocketFlow (in Chinese)](https://www.jiqizhixin.com/articles/2018-09-17-6)


## People

## Venues
- [2018-AAAI](https://aaai.org/Conferences/AAAI-18/wp-content/uploads/2017/12/AAAI-18-Accepted-Paper-List.Web_.pdf)
- [2018-ICLR](https://iclr.cc/Conferences/2018/Schedule)
- [2018-CVPR](http://openaccess.thecvf.com/CVPR2018.py)
- [2018-ICML](https://icml.cc/Conferences/2018/Schedule)
- [2018-ICML Workshop](https://openreview.net/group?id=ICML.cc/2018/ECA): Efficient Credit Assignment in Deep Learning and Reinforcement Learning
- [2018-IJCAI](https://www.ijcai-18.org/accepted-papers/)
- [2018-ECCV](http://openaccess.thecvf.com/ECCV2018.py)
- [2018-BMVC](http://bmvc2018.org/programmedetail.html)
- [2018-NIPS](https://nips.cc/Conferences/2018/Schedule)
- [2018-NIPS Workshop](https://openreview.net/group?id=NIPS.cc/2018/Workshop/CDNNRIA): Compact Deep Neural Network Representation with Industrial Applications
