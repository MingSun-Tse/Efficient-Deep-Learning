# EfficientDNNs
A collection of recent methods on DNN compression and acceleration. There are mainly 5 kinds of methods for efficient DNNs:
- neural architecture re-designing or searching
- pruning (including structured and unstructured)
- quantization
- matrix decomposition
- knowledge distillation

> About abbreviation: In the list below, `o` for oral, `w` for workshop, `s` for spotlight, `b` for best paper.

## Papers
- 2015-INTERSPEECH-[A Diversity-Penalizing Ensemble Training Method for Deep Learning](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_3590.pdf)
- 2015-BMVC-[Data-free parameter pruning for deep neural networks](https://arxiv.org/abs/1507.06149)
- 2015-CVPR-[Learning to generate chairs with convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)
- 2015-CVPR-[Understanding deep image representations by inverting them]()
- 2016-IJCV-[Visualizing deep convolutional neural networks using natural pre-images](https://link.springer.com/content/pdf/10.1007%2Fs11263-016-0911-8.pdf)
- 2016-ICLR-[All you need is a good init](https://arxiv.org/abs/1511.06422) [[code](https://github.com/ducha-aiki/LSUVinit)]
- 2016-ICLR-[Diversity networks](https://pdfs.semanticscholar.org/3f08/1a7d2dbdcd10d71d0340721e4857a73ed7ee.pdf)
- 2016-EMNLP-[Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- 2016-CVPR-[Inverting Visual Representations with Convolutional Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Dosovitskiy_Inverting_Visual_Representations_CVPR_2016_paper.html)
- 2017-CVPR-[All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_All_You_Need_CVPR_2017_paper.html)
- 2017-CVPR-[Learning deep CNN denoiser prior for image restoration](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html)
- 2017-NNs-[Nonredundant sparse feature extraction using autoencoders with receptive fields clustering]()
- 2018-AAAI-[Auto-balanced Filter Pruning for Efficient Convolutional Neural Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16450/16263)
- 2018-AAAI-[Deep Neural Network Compression with Single and Multiple Level Quantization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16479/16742)
- 2018-ICLRo-[Training and Inference with Integers in Deep Neural Networks](https://openreview.net/forum?id=HJGXzmspb)
- 2018-ICLR-[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://openreview.net/forum?id=HJ94fqApW)
- 2018-ICLR-[N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning](https://openreview.net/forum?id=B1hcZZ-AW)
- 2018-ICLR-[Model compression via distillation and quantization](https://openreview.net/forum?id=S1XolQbRW)
- 2018-ICLR-[Towards Image Understanding from Deep Compression Without Decoding](https://openreview.net/forum?id=HkXWCMbRW)
- 2018-ICLR-[Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://openreview.net/forum?id=SkhQHMW0W)
- 2018-ICLR-[Mixed Precision Training of Convolutional Neural Networks using Integer Operations](https://openreview.net/forum?id=H135uzZ0-)
- 2018-ICLR-[Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://openreview.net/forum?id=B1ae1lZRb)
- 2018-ICLR-[Loss-aware Weight Quantization of Deep Networks](https://openreview.net/forum?id=BkrSv0lA-)
- 2018-ICLR-[Alternating Multi-bit Quantization for Recurrent Neural Networks](https://openreview.net/forum?id=S19dR9x0b)
- 2018-ICLR-[Adaptive Quantization of Neural Networks](https://openreview.net/forum?id=SyOK1Sg0W)
- 2018-ICLR-[Variational Network Quantization](https://openreview.net/forum?id=ry-TW-WAb)
- 2018-ICLR-[Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312)
- 2018-ICLRw-[To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression](https://openreview.net/forum?id=Sy1iIDkPM)
- 2018-ICLRw-[Systematic Weight Pruning of DNNs using Alternating Direction Method of Multipliers](https://openreview.net/forum?id=B1_u3cRUG)
- 2018-ICLRw-[Weightless: Lossy weight encoding for deep neural network compression](https://openreview.net/forum?id=rJpXxgaIG)
- 2018-ICLRw-[Variance-based Gradient Compression for Efficient Distributed Deep Learning](https://openreview.net/forum?id=Sy6hd7kvM)
- 2018-ICLRw-[Stacked Filters Stationary Flow For Hardware-Oriented Acceleration Of Deep Convolutional Neural Networks](https://openreview.net/forum?id=HkeAoQQHM)
- 2018-ICLRw-[Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks](https://openreview.net/forum?id=BJbtuRRLM)
- 2018-ICLRw-[Accelerating Neural Architecture Search using Performance Prediction](https://openreview.net/forum?id=HJqk3N1vG)
- 2018-ICLRw-[Nonlinear Acceleration of CNNs](https://openreview.net/forum?id=HkNpF_kDM)
- 2018-CVPR-[Context-Aware Deep Feature Compression for High-Speed Visual Tracking](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Context-Aware_Deep_Feature_CVPR_2018_paper.pdf)
- 2018-CVPR-[NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/pdf/1711.05908.pdf)
- 2018-CVPR-[2018-CVPR-Deep Image Prior](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html) [[code](https://
dmitryulyanov.github.io/deep_image_prior)]
- 2018-ICMLw-[Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures](https://openreview.net/forum?id=SyPicjbWQ)
- 2018-IJCAI-[Efficient DNN Neuron Pruning by Minimizing Layer-wise Nonlinear Reconstruction Error](https://www.ijcai.org/proceedings/2018/0318.pdf)
- 2018-IJCAI-[Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866)
- 2018-IJCAI-[Where to Prune: Using LSTM to Guide End-to-end Pruning]()
- 2018-IJCAI-[Accelerating Convolutional Networks via Global & Dynamic Filter Pruning]()
- 2018-IJCAI-[Optimization based Layer-wise Magnitude-based Pruning for DNN Compression]()
- 2018-IJCAI-[Progressive Blockwise Knowledge Distillation for Neural Network Acceleration]()
- 2018-IJCAI-[Complementary Binary Quantization for Joint Multiple Indexing]()
- 2018-ECCV-[A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf)
- 2018-ECCV-[Coreset-Based Neural Network Compression](http://openaccess.thecvf.com/content_ECCV_2018/papers/Abhimanyu_Dubey_Coreset-Based_Convolutional_Neural_ECCV_2018_paper.pdf)
- 2018-ECCV-[Data-Driven Sparse Structure Selection for Deep Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zehao_Huang_Data-Driven_Sparse_Structure_ECCV_2018_paper.pdf) [[code](https://github.com/TuSimple/sparse-structure-selection)]
- 2018-BMVCo-[Structured Probabilistic Pruning for Convolutional Neural Network Acceleration](http://bmvc2018.org/contents/papers/0870.pdf)
- 2018-BMVC-[Efficient Progressive Neural Architecture Search](http://bmvc2018.org/contents/papers/0291.pdf)
- 2018-NIPS-[Discrimination-aware Channel Pruning for Deep Neural Networks](http://papers.nips.cc/paper/7367-discrimination-aware-channel-pruning-for-deep-neural-networks.pdf)
- 2018-NIPS-[Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](http://papers.nips.cc/paper/7382-frequency-domain-dynamic-pruning-for-convolutional-neural-networks.pdf)
- 2018-NIPS-[ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions](http://papers.nips.cc/paper/7766-channelnets-compact-and-efficient-convolutional-neural-networks-via-channel-wise-convolutions.pdf)
- 2018-NIPS-[DropBlock: A regularization method for convolutional networks](http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks)
- 2019-PR-[Filter-in-Filter: Improve CNNs in a Low-cost Way by Sharing Parameters among the Sub-filters of a Filter](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300640)
- 2018.05-[Compression of Deep Convolutional Neural Networks under Joint Sparsity Constraints](https://arxiv.org/abs/1805.08303)
- 2018.11-[Second-order Optimization Method for Large Mini-batch: Training ResNet-50 on ImageNet in 35 Epochs](https://arxiv.org/abs/1811.12019)
- 2018.11-[Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883) (Kaiming He)


### Papers-Knowledge Distillation
- 1996-[Born again trees](ftp://ftp.stat.berkeley.edu/pub/users/breiman/BAtrees.ps) (proposed compressing neural networks and multipletree predictors by approximating them with a single tree)
- 2006-SIGKDD-[Model compression](https://dl.acm.org/citation.cfm?id=1150464)
- 2010-ML-[A theory of learning from different domains](https://link.springer.com/content/pdf/10.1007%2Fs10994-009-5152-4.pdf)
- 2014-NIPS-[Do deep nets really need to be deep?](https://arxiv.org/abs/1312.6184)
- 2015-[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf) (coined the name "knowledge distillation" and "dark knowledge") [[code](https://github.com/peterliht/knowledge-distillation-pytorch)]
- 2016-ICLR-[Net2net: Accelerating learning via knowledge transfer](https://arxiv.org/abs/1511.05641) (Tianqi Chen and Goodfellow)
- 2015-NIPS-[Bayesian dark knowledge](http://papers.nips.cc/paper/5965-bayesian-dark-knowledge.pdf)
- 2016-ECCV-[Accelerating convolutional neural networks with dominant convolutional kernel and knowledge pre-regression](https://www.researchgate.net/publication/308277663_Accelerating_Convolutional_Neural_Networks_with_Dominant_Convolutional_Kernel_and_Knowledge_Pre-regression)
- 2017-ICLR-[Paying more attention to attention: Improving the performance of convolutional neural networksvia attention transfer](http://arxiv.org/abs/1612.03928)
- 2017-ICLR-[Do deep convolutional nets really need to be deep and convolutional?](https://arxiv.org/pdf/1603.05691.pdf)
- 2017-CVPR-[A gift from knowledge distillation: Fast optimization, network minimization and transfer learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)
- 2017-NIPS-[Sobolev training for neural networks](http://papers.nips.cc/paper/7015-sobolev-training-for-neural-networks.pdf)
- 2017-NIPSw-[Data-Free Knowledge Distillation for Deep Neural Networks](https://arxiv.org/abs/1710.07535) [[code](https://github.com/iRapha/replayed_distillation)]
- 2017.11-[Distilling a Neural Network Into a Soft Decision Tree](https://arxiv.org/abs/1711.09784)
- 2018-AAAI-[DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer](https://arxiv.org/abs/1707.01220)
- 2018-ICML-[Born-Again Neural Networks](https://arxiv.org/pdf/1805.04770.pdf)
- 2018-NIPSw-[Transparent Model Distillation](https://arxiv.org/pdf/1801.08640.pdf)
- 2019-AAAI-[Knowledge Distillation with Adversarial Samples Supporting Decision Boundary](https://arxiv.org/abs/1805.05532)
- 2019-AAAI-[Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://arxiv.org/abs/1811.03233)
- 2017.07-[Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/pdf/1707.01219.pdf)
- 2017.10-[Knowledge Projection for Deep Neural Networks](https://arxiv.org/abs/1710.09505)
- 2017.12-[Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440) (Kaiming He)
- 2018.03-[Interpreting Deep Classifier by Visual Distillation of Dark Knowledge](https://arxiv.org/abs/1803.04042)
- 2018.12-[Learning Student Networks via Feature Embedding](https://arxiv.org/abs/1812.06597)

## People (in alphabeta order)
- [Huang Gao](http://www.gaohuang.net/) @ Tsinghua
- [Mingjie Sun](https://scholar.google.com.vn/citations?user=XVvI7mAAAAAJ&hl=en&oi=ao) @ BUAA
- [Naiyan Wang](http://www.winsty.net/) @ TuSimple
- [Jianguo Li](https://sites.google.com/site/leeplus/) @ Intel
- [Miguel Carreira-Perpinan](https://scholar.google.com/citations?hl=en&user=SYdYhxgAAAAJ) @ UC Merced
- [Song Han](https://songhan.mit.edu/) @ MIT
- [Yihui He](http://yihui-he.github.io/) @ CMU
- [Yang He](https://scholar.google.com.vn/citations?user=vvnFsIIAAAAJ&hl=en&oi=sra) @ University of Technology Sydney 
- [Yunhe Wang](http://www.wangyunhe.site/) @ Huawei
- [Zhuang Liu](https://liuzhuang13.github.io/) @ UC Berkeley

## Venues
- [OpenReview](https://openreview.net/)
- [CVPR & ICCV](http://openaccess.thecvf.com/menu.py)
- [ECCV](https://link.springer.com/conference/eccv)
- [2018-AAAI](https://aaai.org/Conferences/AAAI-18/wp-content/uploads/2017/12/AAAI-18-Accepted-Paper-List.Web_.pdf)
- [2018-ICLR](https://iclr.cc/Conferences/2018/Schedule)
- [2018-ICML](https://icml.cc/Conferences/2018/Schedule)
- [2018-ICML Workshop](https://openreview.net/group?id=ICML.cc/2018/ECA): Efficient Credit Assignment in Deep Learning and Reinforcement Learning
- [2018-IJCAI](https://www.ijcai-18.org/accepted-papers/)
- [2018-BMVC](http://bmvc2018.org/programmedetail.html)
- [2018-NIPS](https://nips.cc/Conferences/2018/Schedule)
- [2018-NIPS Workshop](https://openreview.net/group?id=NIPS.cc/2018/Workshop/CDNNRIA): Compact Deep Neural Network Representation with Industrial Applications
- LLD Workshop: Learning with Limited Data [[1st: 2017 NIPSw](https://lld-workshop.github.io/2017/)] [[2nd: 2019 ICLRw](https://lld-workshop.github.io/)]

## News
- [VALSE 2018年度进展报告 | 深度神经网络加速与压缩 (in Chinese)](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247485375&idx=1&sn=a066fe6f57e6d152719b8815af87e819&chksm=97d7e228a0a06b3e5be18face8716e1ad41d598c2676429a1d174659a384e5f0d5a9be2204ee#rd)
- [机器之心-腾讯AI Lab PocketFlow (in Chinese)](https://www.jiqizhixin.com/articles/2018-09-17-6)