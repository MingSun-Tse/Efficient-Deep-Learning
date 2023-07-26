# EfficientDNNs
A collection of recent methods on DNN compression and acceleration. There are mainly 5 kinds of methods for efficient DNNs:
- neural architecture re-design or search (NAS)
  - maintain accuracy, less cost (e.g., #Params, #FLOPs, etc.): MobileNet, ShuffleNet etc.
  - maintain cost, more accuracy: Inception, ResNeXt, Xception etc.
- pruning (including structured and unstructured)
- quantization
- matrix/low-rank decomposition
- knowledge distillation (KD)

Note, this repo is more about pruning (with lottery ticket hypothesis or LTH as a sub-topic), KD, and quantization. For other topics like NAS, see more comprehensive collections (## Related Repos and Websites) at the end of this file. Welcome to send a pull request if you'd like to add any pertinent papers.

Other repos:
- LTH (lottery ticket hypothesis) and its broader version, *pruning at initialization (PaI)*, now is at the frontier of network pruning. We single out the PaI papers to [this repo](https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization). Welcome to check it out!
- [Awesome-Efficient-ViT](https://github.com/MingSun-Tse/Awesome-Efficient-ViT) for a curated list of efficient vision transformers.

> About abbreviation: In the list below, `o` for oral, `s` for spotlight, `b` for best paper, `w` for workshop.

## Surveys
- 1993-TNN-[Pruning Algorithms -- A survey](https://ieeexplore.ieee.org/abstract/document/248452?casa_token=eJan5NO1DxwAAAAA:chz9BYf22tIO4RHt6nC_x4nbTeTslXiLMrvQElnrXZGbg9fn4c-Alonhq8UYWhT86gXFGO2_)
- 2017-Proceedings of the IEEE-[Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://ieeexplore.ieee.org/document/8114708) [[2020 Book: Efficient Processing of Deep Neural Networks](https://www.morganclaypool.com/doi/pdfplus/10.2200/S01004ED1V01Y202004CAC050?casa_token=rnnqUJmipDEAAAAA:fOs90gKOCbMDqjZlGdc25dCi3H4L0gT1tkEqhNL1eNBpV8h36cvQet9WK0qVRqs9M6irYxbH)]
- 2017.12-[A survey of FPGA-based neural network accelerator](https://arxiv.org/abs/1712.08934)
- 2018-FITEE-[Recent Advances in Efficient Computation of Deep Convolutional Neural Networks](https://link.springer.com/article/10.1631/FITEE.1700789)
- 2018-IEEE Signal Processing Magazine-[Model compression and acceleration for deep neural networks: The principles, progress, and challenges](https://ieeexplore.ieee.org/abstract/document/8253600). [Arxiv extension](https://arxiv.org/abs/1710.09282)
- 2018.8-[A Survey on Methods and Theories of Quantized Neural Networks](https://arxiv.org/abs/1808.04752)
- 2019-JMLR-[Neural Architecture Search: A Survey](http://www.jmlr.org/papers/volume20/18-598/18-598.pdf)
- 2020-MLSys-[What is the state of neural network pruning](https://arxiv.org/abs/2003.03033)
- 2019.02-[The State of Sparsity in Deep Neural Networks](https://arxiv.org/pdf/1902.09574.pdf)
- 2021-TPAMI-[Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks](https://arxiv.org/abs/2004.05937)
- 2021-IJCV-[Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
- 2020-Proceedings of the IEEE-[Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](https://ieeexplore.ieee.org/abstract/document/9043731)
- 2020-Pattern Recognition-[Binary neural networks: A survey](https://www.sciencedirect.com/science/article/pii/S0031320320300856?casa_token=Foe2l0h1AXUAAAAA:z7DaP-QSVCNApUpTsrftp3f2SBfcNj2AH_B0cbzPH4r8BR-cGSns16p1-CQtY7vXuexlPd_Y)
- 2021-TPDS-[The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794)
- 2021-JMLR-[Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554)
- 2022-IJCAI-[Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/abs/2103.06460)
- 2021.6-[Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better](https://arxiv.org/abs/2106.08962)


## Papers [Pruning and Quantization]
**1980s,1990s**
- 1988-NIPS-[A back-propagation algorithm with optimal use of hidden units](https://proceedings.neurips.cc/paper/1988/file/9fc3d7152ba9336a670e36d0ed79bc43-Paper.pdf)
- 1988-NIPS-[Skeletonization: A Technique for Trimming the Fat from a Network via Relevance Assessment](https://papers.nips.cc/paper/1988/file/07e1cd7dca89a1678042477183b7ac3f-Paper.pdf)
- 1988-NIPS-[What Size Net Gives Valid Generalization?](https://papers.nips.cc/paper/1988/file/1d7f7abc18fcb43975065399b0d1e48e-Paper.pdf)
- 1989-NIPS-[Dynamic Behavior of Constained Back-Propagation Networks](https://proceedings.neurips.cc/paper/1989/hash/85d8ce590ad8981ca2c8286f79f59954-Abstract.html)
- 1988-NIPS-[Comparing Biases for Minimal Network Construction with Back-Propagation](https://papers.nips.cc/paper/1988/file/1c9ac0159c94d8d0cbedc973445af2da-Paper.pdf)
- 1989-NIPS-[Optimal Brain Damage](https://papers.nips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)
- 1990-NN-[A simple procedure for pruning back-propagation trained neural networks](https://ieeexplore.ieee.org/abstract/document/80236)
- 1993-ICNN-[Optimal Brain Surgeon and general network pruning](https://ieeexplore.ieee.org/abstract/document/298572?casa_token=8a8fUVuadHEAAAAA:tgRbetEERx1Bdh6RCa27mok9SAPNc8Y33qy2ScdTNOCs_ajHlaUv4_nnvDNJp3jZbb13uouD)

**2000s**
- 2001-JMLR-[Sparse Bayesian learning and the relevance vector machine](https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf)
- 2007-Book-[The minimum description length principle]()

**2011**
- 2011-JMLR-[Learning with Structured Sparsity](http://www.jmlr.org/papers/v12/huang11b.html)
- 2011-NIPSw-[Improving the speed of neural networks on CPUs](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.308.2766)

**2013**
- 2013-NIPS-[Predicting Parameters in Deep Learning](http://papers.nips.cc/paper/5025-predicting-parameters-in-deep-learning)
- 2013.08-[Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432)

**2014**
- 2014-BMVC-[Speeding up convolutional neural networks with low rank expansions](https://arxiv.org/abs/1405.3866)
- 2014-INTERSPEECH-[1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs](https://www.isca-speech.org/archive/interspeech_2014/i14_1058.html)
- 2014-NIPS-[Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](http://papers.nips.cc/paper/5544-exploiting-linear-structure-within-convolutional-networks-for-efficient-evaluation)
- 2014-NIPS-[Do deep neural nets really need to be deep](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep)
- 2014.12-[Memory bounded deep convolutional networks](https://arxiv.org/abs/1412.1442)
  
**2015**
- 2015-ICLR-[Speeding-up convolutional neural networks using fine-tuned cp-decomposition](https://arxiv.org/abs/1412.6553)
- 2015-ICML-[Compressing neural networks with the hashing trick](http://proceedings.mlr.press/v37/chenc15.pdf)
- 2015-INTERSPEECH-[A Diversity-Penalizing Ensemble Training Method for Deep Learning](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_3590.pdf)
- 2015-BMVC-[Data-free parameter pruning for deep neural networks](https://arxiv.org/abs/1507.06149)
- 2015-BMVC-[Learning the structure of deep architectures using l1 regularization](http://www.bmva.org/bmvc/2015/papers/paper023/paper023.pdf)
- 2015-NIPS-[Learning both Weights and Connections for Efficient Neural Network](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network)
- 2015-NIPS-[Binaryconnect: Training deep neural networks with binary weights during propagations](http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b)
- 2015-NIPS-[Structured Transforms for Small-Footprint Deep Learning](http://papers.nips.cc/paper/5869-structured-transforms-for-small-footprint-deep-learning)
- 2015-NIPS-[Tensorizing Neural Networks](http://papers.nips.cc/paper/5787-tensorizing-neural-networks)
- 2015-NIPSw-[Distilling Intractable Generative Models](http://homepages.inf.ed.ac.uk/s1459647/papers/distilling_generative_models.pdf)
- 2015-NIPSw-[Federated Optimization:Distributed Optimization Beyond the Datacenter](https://arxiv.org/abs/1511.03575)
- 2015-CVPR-[Efficient and Accurate Approximations of Nonlinear Convolutional Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Zhang_Efficient_and_Accurate_2015_CVPR_paper.html) [2016 TPAMI version: [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://ieeexplore.ieee.org/abstract/document/7332968)]
- 2015-CVPR-[Sparse Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2015/html/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.html)
- 2015-ICCV-[An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Cheng_An_Exploration_of_ICCV_2015_paper.html)
- 2015.12-[Exploiting Local Structures with the Kronecker Layer in Convolutional Networks](https://arxiv.org/abs/1512.09194)

**2016**
- 2016-ICLR-[Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) [Best paper!]
- 2016-ICLR-[All you need is a good init](https://arxiv.org/abs/1511.06422) [[Code](https://github.com/ducha-aiki/LSUVinit)]
- 2016-ICLR-[Data-dependent Initializations of Convolutional Neural Networks](https://arxiv.org/abs/1511.06856) [[Code](https://github.com/philkr/magic_init)]
- 2016-ICLR-[Convolutional neural networks with low-rank regularization](https://arxiv.org/abs/1511.06067) [[Code](https://github.com/chengtaipu/lowrankcnn)]
- 2016-ICLR-[Diversity networks](https://pdfs.semanticscholar.org/3f08/1a7d2dbdcd10d71d0340721e4857a73ed7ee.pdf)
- 2016-ICLR-[Neural networks with few multiplications](https://arxiv.org/abs/1510.03009)
- 2016-ICLR-[Compression of deep convolutional neural networks for fast and low power mobile applications](https://arxiv.org/abs/1511.06530)
- 2016-ICLRw-[Randomout: Using a convolutional gradient norm to win the filter lottery](https://arxiv.org/abs/1602.05931)
- 2016-CVPR-[Fast algorithms for convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Lavin_Fast_Algorithms_for_CVPR_2016_paper.html)
- 2016-CVPR-[Fast ConvNets Using Group-wise Brain Damage](http://openaccess.thecvf.com/content_cvpr_2016/html/Lebedev_Fast_ConvNets_Using_CVPR_2016_paper.html)
- 2016-BMVC-[Learning neural network architectures using backpropagation](https://arxiv.org/abs/1511.05497)
- 2016-ECCV-[Less is more: Towards compact cnns](http://users.umiacs.umd.edu/~hzhou/paper/zhou_ECCV2016.pdf)
- 2016-EMNLP-[Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- 2016-NIPS-[Learning Structured Sparsity in Deep Neural Networks](https://proceedings.neurips.cc/paper/2016/hash/41bfd20a38bb1b0bec75acf0845530a7-Abstract.html) [[Caffe Code](https://github.com/wenwei202/caffe)]
- 2016-NIPS-[Dynamic Network Surgery for Efficient DNNs](http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns) [[Caffe Code](https://github.com/yiwenguo/Dynamic-Network-Surgery)]
- 2016-NIPS-[Learning the Number of Neurons in Deep Neural Networks](http://papers.nips.cc/paper/6372-learning-the-number-of-neurons-in-deep-networks)
- 2016-NIPS-[Memory-Efficient Backpropagation Through Time](http://papers.nips.cc/paper/6220-memory-efficient-backpropagation-through-time)
- 2016-NIPS-[PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions](http://papers.nips.cc/paper/6463-perforatedcnns-acceleration-through-elimination-of-redundant-convolutions)
- 2016-NIPS-[LightRNN: Memory and Computation-Efficient Recurrent Neural Networks](http://papers.nips.cc/paper/6512-lightrnn-memory-and-computation-efficient-recurrent-neural-networks)
- 2016-NIPS-[CNNpack: packing convolutional neural networks in the frequency domain](https://papers.nips.cc/paper/6390-cnnpack-packing-convolutional-neural-networks-in-the-frequency-domain.pdf)
- 2016-ISCA-[Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks](https://people.csail.mit.edu/emer/papers/2016.06.isca.eyeriss_architecture.pdf)
- 2016-ICASSP-[Learning compact recurrent neural networks](https://arxiv.org/abs/1604.02594)
- 2016-CoNLL-[Compression of Neural Machine Translation Models via Pruning](https://arxiv.org/abs/1606.09274)
- 2016.03-[Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)
- 2016.06-[Structured Convolution Matrices for Energy-efficient Deep learning](https://arxiv.org/abs/1606.02407)
- 2016.06-[Deep neural networks are robust to weight binarization and other non-linear distortions](https://arxiv.org/abs/1606.01981)
- 2016.06-[Hypernetworks](https://arxiv.org/abs/1609.09106)
- 2016.07-IHT-[Training skinny deep neural networks with iterative hard thresholding methods](https://arxiv.org/abs/1607.05423)
- 2016.08-[Recurrent Neural Networks With Limited Numerical Precision](https://arxiv.org/abs/1608.06902)
- 2016.10-[Deep model compression: Distilling knowledge from noisy teachers](https://arxiv.org/abs/1610.09650)
- 2016.10-[Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/abs/1610.02527)
- 2016.11-[Alternating Direction Method of Multipliers for Sparse Convolutional Neural Networks](https://arxiv.org/abs/1611.01590)

**2017**
- 2017-ICLR-[Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) [[PyTorch Reimpl. #1](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning)] [[PyTorch Reimpl. #2](https://github.com/MingSun-Tse/Regularization-Pruning)]
- 2017-ICLR-[Pruning Convolutional Neural Networks for Resource Efficient Inference](https://openreview.net/forum?id=SJGCiw5gl&noteId=SJGCiw5gl)
- 2017-ICLR-[Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights](https://arxiv.org/abs/1702.03044) [[Code](https://github.com/Mxbonn/INQ-pytorch)]
- 2017-ICLR-[Do Deep Convolutional Nets Really Need to be Deep and Convolutional?](https://arxiv.org/abs/1603.05691)
- 2017-ICLR-[DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
- 2017-ICLR-[Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)
- 2017-ICLR-[Towards the Limit of Network Quantization](https://openreview.net/forum?id=rJ8uNptgl)
- 2017-ICLR-[Loss-aware Binarization of Deep Networks](https://openreview.net/forum?id=S1oWlN9ll&noteId=S1oWlN9ll)
- 2017-ICLR-[Trained Ternary Quantization](https://openreview.net/forum?id=S1_pAu9xl&noteId=S1_pAu9xl) [[Code](https://github.com/czhu95/ternarynet)]
- 2017-ICLR-[Exploring Sparsity in Recurrent Neural Networks](https://openreview.net/forum?id=BylSPv9gx&noteId=BylSPv9gx)
- 2017-ICLR-[Soft Weight-Sharing for Neural Network Compression](https://openreview.net/forum?id=HJGwcKclx) [[Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/5u7h3l/r_compressing_nn_with_shannons_blessing/)] [[Code](https://github.com/KarenUllrich/Tutorial-SoftWeightSharingForNNCompression)]
- 2017-ICLR-[Variable Computation in Recurrent Neural Networks](https://openreview.net/forum?id=S1LVSrcge&noteId=S1LVSrcge)
- 2017-ICLR-[Training Compressed Fully-Connected Networks with a Density-Diversity Penalty](https://openreview.net/forum?id=Hku9NK5lx)
- 2017-ICML-[Theoretical Properties for Neural Networks with Weight Matrices of Low Displacement Rank](https://arxiv.org/abs/1703.00144)
- 2017-ICML-[Deep Tensor Convolution on Multicores](http://proceedings.mlr.press/v70/budden17a.html)
- 2017-ICML-[Delta Networks for Optimized Recurrent Network Computation](http://proceedings.mlr.press/v70/neil17a.html)
- 2017-ICML-[Beyond Filters: Compact Feature Map for Portable Deep Model](http://proceedings.mlr.press/v70/wang17m.html)
- 2017-ICML-[Combined Group and Exclusive Sparsity for Deep Neural Networks](http://proceedings.mlr.press/v70/yoon17a.html)
- 2017-ICML-[MEC: Memory-efficient Convolution for Deep Neural Network](http://proceedings.mlr.press/v70/cho17a.html)
- 2017-ICML-[Deciding How to Decide: Dynamic Routing in Artificial Neural Networks](http://proceedings.mlr.press/v70/mcgill17a.html)
- 2017-ICML-[ZipML: Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning](http://proceedings.mlr.press/v70/zhang17e.html)
- 2017-ICML-[Analytical Guarantees on Numerical Precision of Deep Neural Networks](http://proceedings.mlr.press/v70/sakr17a.html)
- 2017-ICML-[Adaptive Neural Networks for Efficient Inference](http://proceedings.mlr.press/v70/bolukbasi17a.html)
- 2017-ICML-[SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization](http://proceedings.mlr.press/v70/kim17b.html)
- 2017-CVPR-[Learning deep CNN denoiser prior for image restoration](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html)
- 2017-CVPR-[Deep roots: Improving cnn efficiency with hierarchical filter groups](http://openaccess.thecvf.com/content_cvpr_2017/html/Ioannou_Deep_Roots_Improving_CVPR_2017_paper.html)
- 2017-CVPR-[More is less: A more complicated network with less inference complexity](http://openaccess.thecvf.com/content_cvpr_2017/html/Dong_More_Is_Less_CVPR_2017_paper.html) [[PyTorch Code](https://github.com/D-X-Y/DXY-Projects/tree/master/LCCL)]
- 2017-CVPR-[All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_All_You_Need_CVPR_2017_paper.html)
- 2017-CVPR-ResNeXt-[Aggregated Residual Transformations for Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html)
- 2017-CVPR-[Xception: Deep learning with depthwise separable convolutions](http://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html)
- 2017-CVPR-[Designing Energy-Efficient CNN using Energy-aware Pruning](http://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html)
- 2017-CVPR-[Spatially Adaptive Computation Time for Residual Networks](http://openaccess.thecvf.com/content_cvpr_2017/html/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.html)
- 2017-CVPR-[Network Sketching: Exploiting Binary Structure in Deep CNNs](http://openaccess.thecvf.com/content_cvpr_2017/html/Guo_Network_Sketching_Exploiting_CVPR_2017_paper.html)
- 2017-CVPR-[A Compact DNN: Approaching GoogLeNet-Level Accuracy of Classification and Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2017/html/Wu_A_Compact_DNN_CVPR_2017_paper.html)
- 2017-ICCV-[Channel pruning for accelerating very deep neural networks](http://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) [[Caffe Code](https://github.com/yihui-he/channel-pruning)]
- 2017-ICCV-[Learning efficient convolutional networks through network slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) [[PyTorch Code](https://github.com/liuzhuang13/slimming/)]
- 2017-ICCV-[ThiNet: A filter level pruning method for deep neural network compression](http://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html) [[Project](http://lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html)] [[Caffe Code](https://github.com/Roll920/ThiNet_Code)] [[2018 TPAMI version](https://ieeexplore.ieee.org/document/8416559)]
- 2017-ICCV-[Interleaved group convolutions](http://openaccess.thecvf.com/content_iccv_2017/html/Zhang_Interleaved_Group_Convolutions_ICCV_2017_paper.html)
- 2017-ICCV-[Coordinating Filters for Faster Deep Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Wen_Coordinating_Filters_for_ICCV_2017_paper.html) [[Caffe Code](https://github.com/wenwei202/caffe)]
- 2017-ICCV-[Performance Guaranteed Network Acceleration via High-Order Residual Quantization](http://openaccess.thecvf.com/content_iccv_2017/html/Li_Performance_Guaranteed_Network_ICCV_2017_paper.html)
- 2017-NIPS-[Net-trim: Convex pruning of deep neural networks with performance guarantee](http://papers.nips.cc/paper/6910-net-trim-convex-pruning-of-deep-neural-networks-with-performance-guarantee) [[Code](https://github.com/DNNToolBox/Net-Trim)] (Journal version: [2020-SIAM-Fast Convex Pruning of Deep Neural Networks](https://epubs.siam.org/doi/abs/10.1137/19M1246468))
- 2017-NIPS-[Runtime neural pruning](http://papers.nips.cc/paper/6813-runtime-neural-pruning)
- 2017-NIPS-[Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](http://papers.nips.cc/paper/7071-learning-to-prune-deep-neural-networks-via-layer-wise-optimal-brain-surgeon) [[Code](https://github.com/csyhhu/L-OBS)]
- 2017-NIPS-[Federated Multi-Task Learning](http://papers.nips.cc/paper/7029-federated-multi-task-learning)
- 2017-NIPS-[Towards Accurate Binary Convolutional Neural Network](http://papers.nips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network)
- 2017-NIPS-[Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations](http://papers.nips.cc/paper/6714-soft-to-hard-vector-quantization-for-end-to-end-learning-compressible-representations)
- 2017-NIPS-[TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](http://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning)
- 2017-NIPS-[Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks](http://papers.nips.cc/paper/6771-flexpoint-an-adaptive-numerical-format-for-efficient-training-of-deep-neural-networks)
- 2017-NIPS-[Training Quantized Nets: A Deeper Understanding](http://papers.nips.cc/paper/7163-training-quantized-nets-a-deeper-understanding)
- 2017-NIPS-[The Reversible Residual Network: Backpropagation Without Storing Activations](http://papers.nips.cc/paper/6816-the-reversible-residual-network-backpropagation-without-storing-activations) [[Code](https://github.com/renmengye/revnet-public)]
- 2017-NIPS-[Compression-aware Training of Deep Networks](http://papers.nips.cc/paper/6687-compression-aware-training-of-deep-networks)
- 2017-FPGA-[ESE: efficient speech recognition engine with compressed LSTM on FPGA](https://pdfs.semanticscholar.org/99d2/07c18ba48e41560f3081ea1b7c6fde98c1ce.pdf) [Best paper!]
- 2017-AISTATS-[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- 2017-ICASSP-[Accelerating Deep Convolutional Networks using low-precision and sparsity](https://arxiv.org/abs/1610.00324)
- 2017-NNs-[Nonredundant sparse feature extraction using autoencoders with receptive fields clustering](https://www.sciencedirect.com/science/article/pii/S0893608017300928)
- 2017.02-[The Power of Sparsity in Convolutional Neural Networks](https://arxiv.org/abs/1702.06257)
- 2017.07-[Stochastic, Distributed and Federated Optimization for Machine Learning](https://arxiv.org/abs/1707.01155)
- 2017.05-[Structural Compression of Convolutional Neural Networks Based on Greedy Filter Pruning](https://arxiv.org/abs/1705.07356)
- 2017.07-[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/abs/1707.09870)
- 2017.11-[GPU Kernels for Block-Sparse Weights](https://openai.com/blog/block-sparse-gpu-kernels/) [[Code](https://github.com/openai/blocksparse)] (OpenAI)
- 2017.11-[Block-sparse recurrent neural networks](https://arxiv.org/abs/1711.02782)

**2018**
- 2018-AAAI-[Auto-balanced Filter Pruning for Efficient Convolutional Neural Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16450/16263)
- 2018-AAAI-[Deep Neural Network Compression with Single and Multiple Level Quantization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16479/16742)
- 2018-AAAI-[Dynamic Deep Neural Networks_Optimizing Accuracy-Efficiency Trade-offs by Selective Execution](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16291)
- 2018-ICLRo-[Training and Inference with Integers in Deep Neural Networks](https://openreview.net/forum?id=HJGXzmspb)
- 2018-ICLR-[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://openreview.net/forum?id=HJ94fqApW)
- 2018-ICLR-[N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning](https://openreview.net/forum?id=B1hcZZ-AW)
- 2018-ICLR-[Model compression via distillation and quantization](https://openreview.net/forum?id=S1XolQbRW)
- 2018-ICLR-[Towards Image Understanding from Deep Compression Without Decoding](https://openreview.net/forum?id=HkXWCMbRW)
- 2018-ICLR-[Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://openreview.net/forum?id=SkhQHMW0W)
- 2018-ICLR-[Mixed Precision Training of Convolutional Neural Networks using Integer Operations](https://openreview.net/forum?id=H135uzZ0-)
- 2018-ICLR-[Mixed Precision Training](https://openreview.net/forum?id=r1gs9JgRZ)
- 2018-ICLR-[Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://openreview.net/forum?id=B1ae1lZRb)
- 2018-ICLR-[Loss-aware Weight Quantization of Deep Networks](https://openreview.net/forum?id=BkrSv0lA-)
- 2018-ICLR-[Alternating Multi-bit Quantization for Recurrent Neural Networks](https://openreview.net/forum?id=S19dR9x0b)
- 2018-ICLR-[Adaptive Quantization of Neural Networks](https://openreview.net/forum?id=SyOK1Sg0W)
- 2018-ICLR-[Variational Network Quantization](https://openreview.net/forum?id=ry-TW-WAb)
- 2018-ICLR-[Espresso: Efficient Forward Propagation for Binary Deep Neural Networks](https://openreview.net/forum?id=Sk6fD5yCb&noteId=Sk6fD5yCb)
- 2018-ICLR-[Learning to share: Simultaneous parameter tying and sparsification in deep learning](https://openreview.net/forum?id=rypT3fb0b&noteId=rkwxPE67M)
- 2018-ICLR-[Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312)
- 2018-ICLR-[WRPN: Wide Reduced-Precision Networks](https://openreview.net/forum?id=B1ZvaaeAZ&noteId=B1ZvaaeAZ)
- 2018-ICLR-[Deep rewiring: Training very sparse deep networks](https://openreview.net/forum?id=BJ_wN01C-&noteId=BJ_wN01C-)
- 2018-ICLR-[Efficient sparse-winograd convolutional neural networks](https://arxiv.org/pdf/1802.06367.pdf) [[Code](https://github.com/xingyul/Sparse-Winograd-CNN)]
- 2018-ICLR-[Learning Intrinsic Sparse Structures within Long Short-term Memory](https://arxiv.org/pdf/1709.05027)
- 2018-ICLR-[Multi-scale dense networks for resource efficient image classification](https://arxiv.org/abs/1703.09844)
- 2018-ICLR-[Compressing Word Embedding via Deep Compositional Code Learning](https://openreview.net/forum?id=BJRZzFlRb&noteId=BJRZzFlRb)
- 2018-ICLR-[Learning Discrete Weights Using the Local Reparameterization Trick](https://openreview.net/forum?id=BySRH6CpW)
- 2018-ICLR-[Training wide residual networks for deployment using a single bit for each weight](https://openreview.net/forum?id=rytNfI1AZ&noteId=rytNfI1AZ)
- 2018-ICLR-[The High-Dimensional Geometry of Binary Neural Networks](https://openreview.net/forum?id=B1IDRdeCW&noteId=B1IDRdeCW)
- 2018-ICLRw-[To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression](https://openreview.net/forum?id=Sy1iIDkPM) (Similar topic: [2018-NIPSw-nip in the bud](https://openreview.net/forum?id=r1lbgwFj5m), [2018-NIPSw-rethink](https://openreview.net/forum?id=r1eLk2mKiX))
- 2018-CVPR-[Context-Aware Deep Feature Compression for High-Speed Visual Tracking](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Context-Aware_Deep_Feature_CVPR_2018_paper.pdf)
- 2018-CVPR-[NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/pdf/1711.05908.pdf)
- 2018-CVPR-[Condensenet: An efficient densenet using learned group convolutions](http://openaccess.thecvf.com/content_cvpr_2018/html/Huang_CondenseNet_An_Efficient_CVPR_2018_paper.html) [[Code](https://github.com/ShichenLiu/CondenseNet)]
- 2018-CVPR-[Shift: A zero flop, zero parameter alternative to spatial convolutions](http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Shift_A_Zero_CVPR_2018_paper.html)
- 2018-CVPR-[Explicit Loss-Error-Aware Quantization for Low-Bit Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Explicit_Loss-Error-Aware_Quantization_CVPR_2018_paper.html)
- 2018-CVPR-[Interleaved structured sparse convolutional neural networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Xie_Interleaved_Structured_Sparse_CVPR_2018_paper.html)
- 2018-CVPR-[Towards Effective Low-bitwidth Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhuang_Towards_Effective_Low-Bitwidth_CVPR_2018_paper.pdf)
- 2018-CVPR-[CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](http://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)
- 2018-CVPR-[Blockdrop: Dynamic inference paths in residual networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_BlockDrop_Dynamic_Inference_CVPR_2018_paper.html)
- 2018-CVPR-[Nestednet: Learning nested sparse structures in deep neural networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Kim_NestedNet_Learning_Nested_CVPR_2018_paper.html)
- 2018-CVPR-[Stochastic downsampling for cost-adjustable inference and improved regularization in convolutional networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Kuen_Stochastic_Downsampling_for_CVPR_2018_paper.html)
- 2018-CVPR-[Wide Compression: Tensor Ring Nets](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Wide_Compression_Tensor_CVPR_2018_paper.html)
- 2018-CVPR-[Learning Compact Recurrent Neural Networks With Block-Term Tensor Decomposition](http://openaccess.thecvf.com/content_cvpr_2018/html/Ye_Learning_Compact_Recurrent_CVPR_2018_paper.html)
- 2018-CVPR-[Learning Time/Memory-Efficient Deep Architectures With Budgeted Super Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Veniat_Learning_TimeMemory-Efficient_Deep_CVPR_2018_paper.html)
- 2018-CVPR-[HydraNets: Specialized Dynamic Architectures for Efficient Inference](http://openaccess.thecvf.com/content_cvpr_2018/html/Mullapudi_HydraNets_Specialized_Dynamic_CVPR_2018_paper.html)
- 2018-CVPR-[SYQ: Learning Symmetric Quantization for Efficient Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Faraone_SYQ_Learning_Symmetric_CVPR_2018_paper.html)
- 2018-CVPR-[Towards Effective Low-Bitwidth Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhuang_Towards_Effective_Low-Bitwidth_CVPR_2018_paper.html)
- 2018-CVPR-[Two-Step Quantization for Low-Bit Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Two-Step_Quantization_for_CVPR_2018_paper.html)
- 2018-CVPR-[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)
- 2018-CVPR-["Learning-Compression" Algorithms for Neural Net Pruning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Carreira-Perpinan_Learning-Compression_Algorithms_for_CVPR_2018_paper.pdf)
- 2018-CVPR-[PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/pdf/1711.05769v2.pdf) [[Code](https://github.com/arunmallya/packnet)]
- 2018-CVPR-[MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Gordon_MorphNet_Fast__CVPR_2018_paper.html) [[Code](https://github.com/google-research/morph-net)]
- 2018-CVPR-[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)
- 2018-CVPRw-[Squeezenext: Hardware-aware neural network design](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w33/html/Gholami_SqueezeNext_Hardware-Aware_Neural_CVPR_2018_paper.html)
- 2018-IJCAI-[Efficient DNN Neuron Pruning by Minimizing Layer-wise Nonlinear Reconstruction Error](https://www.ijcai.org/proceedings/2018/0318.pdf)
- 2018-IJCAI-[Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866) [[PyTorch Code](https://github.com/he-y/soft-filter-pruning)]
- 2018-IJCAI-[Where to Prune: Using LSTM to Guide End-to-end Pruning](https://www.ijcai.org/proceedings/2018/0445.pdf)
- 2018-IJCAI-[Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf)
- 2018-IJCAI-[Optimization based Layer-wise Magnitude-based Pruning for DNN Compression](http://staff.ustc.edu.cn/~chaoqian/ijcai18-olmp.pdf)
- 2018-IJCAI-[Progressive Blockwise Knowledge Distillation for Neural Network Acceleration](https://www.ijcai.org/proceedings/2018/0384.pdf)
- 2018-IJCAI-[Complementary Binary Quantization for Joint Multiple Indexing](https://www.ijcai.org/proceedings/2018/0292.pdf)
- 2018-ICML-[Compressing Neural Networks using the Variational Information Bottleneck](http://proceedings.mlr.press/v80/dai18d.html)
- 2018-ICML-[DCFNet: Deep Neural Network with Decomposed Convolutional Filters](http://proceedings.mlr.press/v80/qiu18a.html)
- 2018-ICML-[Deep k-Means Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions](http://proceedings.mlr.press/v80/wu18h.html)
- 2018-ICML-[Error Compensated Quantized SGD and its Applications to Large-scale Distributed Optimization](http://proceedings.mlr.press/v80/wu18d.html)
- 2018-ICML-[High Performance Zero-Memory Overhead Direct Convolutions](http://proceedings.mlr.press/v80/zhang18d.html)
- 2018-ICML-[Kronecker Recurrent Units](http://proceedings.mlr.press/v80/jose18a.html)
- 2018-ICML-[Weightless: Lossy weight encoding for deep neural network compression](http://proceedings.mlr.press/v80/reagan18a.html)
- 2018-ICML-[StrassenNets: Deep learning with a multiplication budget](http://proceedings.mlr.press/v80/tschannen18a.html)
- 2018-ICML-[Learning Compact Neural Networks with Regularization](http://proceedings.mlr.press/v80/oymak18a.html)
- 2018-ICML-[WSNet: Compact and Efficient Networks Through Weight Sampling](http://proceedings.mlr.press/v80/jin18d.html)
- 2018-ICML-[Gradually Updated Neural Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1711.09280) [[Code](https://github.com/joe-siyuan-qiao/GUNN)]
- 2018-ICML-[On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization](https://arxiv.org/abs/1802.06509)
- 2018-ICML-[Understanding and simplifying one-shot architecture search](http://proceedings.mlr.press/v80/bender18a.html)
- 2018-ECCV-[A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf) [[Code](https://github.com/KaiqiZhang/admm-pruning)]
- 2018-ECCV-[Coreset-Based Neural Network Compression](http://openaccess.thecvf.com/content_ECCV_2018/papers/Abhimanyu_Dubey_Coreset-Based_Convolutional_Neural_ECCV_2018_paper.pdf)
- 2018-ECCV-[Data-Driven Sparse Structure Selection for Deep Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zehao_Huang_Data-Driven_Sparse_Structure_ECCV_2018_paper.pdf) [[MXNet Code](https://github.com/TuSimple/sparse-structure-selection)]
- 2018-ECCV-[Training Binary Weight Networks via Semi-Binary Decomposition](http://openaccess.thecvf.com/content_ECCV_2018/html/Qinghao_Hu_Training_Binary_Weight_ECCV_2018_paper.html)
- 2018-ECCV-[Learning Compression from Limited Unlabeled Data](http://openaccess.thecvf.com/content_ECCV_2018/html/Xiangyu_He_Learning_Compression_from_ECCV_2018_paper.html)
- 2018-ECCV-[Constraint-Aware Deep Neural Network Compression](http://openaccess.thecvf.com/content_ECCV_2018/html/Changan_Chen_Constraints_Matter_in_ECCV_2018_paper.html)
- 2018-ECCV-[Sparsely Aggregated Convolutional Networks](http://openaccess.thecvf.com/content_ECCV_2018/html/Ligeng_Zhu_Sparsely_Aggregated_Convolutional_ECCV_2018_paper.html)
- 2018-ECCV-[Deep Expander Networks: Efficient Deep Networks from Graph Theory](http://openaccess.thecvf.com/content_ECCV_2018/html/Ameya_Prabhu_Deep_Expander_Networks_ECCV_2018_paper.html) [[Code](https://github.com/DrImpossible/Deep-Expander-Networks)]
- 2018-ECCV-[SparseNet-Sparsely Aggregated Convolutional Networks](https://arxiv.org/abs/1801.05895) [[Code](https://github.com/Lyken17/SparseNet)]
- 2018-ECCV-[Ask, acquire, and attack: Data-free uap generation using class impressions](http://openaccess.thecvf.com/content_ECCV_2018/html/Konda_Reddy_Mopuri_Ask_Acquire_and_ECCV_2018_paper.html)
- 2018-ECCV-[Netadapt: Platform-aware neural network adaptation for mobile applications](http://openaccess.thecvf.com/content_ECCV_2018/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.html)
- 2018-ECCV-[Clustering Convolutional Kernels to Compress Deep Neural Networks](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01237-3_14.pdf)
- 2018-ECCV-[Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](http://openaccess.thecvf.com/content_ECCV_2018/html/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.html)
- 2018-ECCV-[Extreme Network Compression via Filter Group Approximation](http://openaccess.thecvf.com/content_ECCV_2018/html/Bo_Peng_Extreme_Network_Compression_ECCV_2018_paper.html)
- 2018-ECCV-[Convolutional Networks with Adaptive Inference Graphs](http://openaccess.thecvf.com/content_ECCV_2018/html/Andreas_Veit_Convolutional_Networks_with_ECCV_2018_paper.html)
- 2018-ECCV-[SkipNet: Learning Dynamic Routing in Convolutional Networks](https://arxiv.org/abs/1711.09485) [[Code](https://github.com/ucbdrive/skipnet)]
- 2018-ECCV-[Value-aware Quantization for Training and Inference of Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/html/Eunhyeok_Park_Value-aware_Quantization_for_ECCV_2018_paper.html)
- 2018-ECCV-[LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html)
- 2018-ECCV-[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](http://openaccess.thecvf.com/content_ECCV_2018/html/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.html)
- 2018-ECCV-[Piggyback: Adapting a single network to multiple tasks by learning to mask weights](http://openaccess.thecvf.com/content_ECCV_2018/html/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.html)
- 2018-BMVCo-[Structured Probabilistic Pruning for Convolutional Neural Network Acceleration](http://bmvc2018.org/contents/papers/0870.pdf)
- 2018-BMVC-[Efficient Progressive Neural Architecture Search](http://bmvc2018.org/contents/papers/0291.pdf)
- 2018-BMVC-[Igcv3: Interleaved lowrank group convolutions for efficient deep neural networks](https://arxiv.org/abs/1806.00178)
- 2018-NIPS-[Discrimination-aware Channel Pruning for Deep Neural Networks](http://papers.nips.cc/paper/7367-discrimination-aware-channel-pruning-for-deep-neural-networks.pdf)
- 2018-NIPS-[Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](http://papers.nips.cc/paper/7382-frequency-domain-dynamic-pruning-for-convolutional-neural-networks.pdf)
- 2018-NIPS-[ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions](http://papers.nips.cc/paper/7766-channelnets-compact-and-efficient-convolutional-neural-networks-via-channel-wise-convolutions.pdf)
- 2018-NIPS-[DropBlock: A regularization method for convolutional networks](http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks)
- 2018-NIPS-[Constructing fast network through deconstruction of convolution](http://papers.nips.cc/paper/7835-constructing-fast-network-through-deconstruction-of-convolution)
- 2018-NIPS-[Learning Versatile Filters for Efficient Convolutional Neural Networks](https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks) [[Code](https://github.com/NoahEC/Versatile-Filters)]
- 2018-NIPS-[Moonshine: Distilling with cheap convolutions](http://papers.nips.cc/paper/7553-moonshine-distilling-with-cheap-convolutions)
- 2018-NIPS-[HitNet: hybrid ternary recurrent neural network](http://papers.nips.cc/paper/7341-hitnet-hybrid-ternary-recurrent-neural-network)
- 2018-NIPS-[FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network](http://papers.nips.cc/paper/8116-fastgrnn-a-fast-accurate-stable-and-tiny-kilobyte-sized-gated-recurrent-neural-network)
- 2018-NIPS-[Training DNNs with Hybrid Block Floating Point](http://papers.nips.cc/paper/7327-training-dnns-with-hybrid-block-floating-point)
- 2018-NIPS-[Reversible Recurrent Neural Networks](http://papers.nips.cc/paper/8117-reversible-recurrent-neural-networks)
- 2018-NIPS-[Synaptic Strength For Convolutional Neural Network](http://papers.nips.cc/paper/8218-synaptic-strength-for-convolutional-neural-network)
- 2018-NIPS-[Learning sparse neural networks via sensitivity-driven regularization](http://papers.nips.cc/paper/7644-learning-sparse-neural-networks-via-sensitivity-driven-regularization)
- 2018-NIPS-[Multi-Task Zipping via Layer-wise Neuron Sharing](http://papers.nips.cc/paper/7841-multi-task-zipping-via-layer-wise-neuron-sharing)
- 2018-NIPS-[A Linear Speedup Analysis of Distributed Deep Learning with Sparse and Quantized Communication](http://papers.nips.cc/paper/7519-a-linear-speedup-analysis-of-distributed-deep-learning-with-sparse-and-quantized-communication)
- 2018-NIPS-[Gradient Sparsification for Communication-Efficient Distributed Optimization](http://papers.nips.cc/paper/7405-gradient-sparsification-for-communication-efficient-distributed-optimization)
- 2018-NIPS-[GradiVeQ: Vector Quantization for Bandwidth-Efficient Gradient Aggregation in Distributed CNN Training](http://papers.nips.cc/paper/7759-gradiveq-vector-quantization-for-bandwidth-efficient-gradient-aggregation-in-distributed-cnn-training)
- 2018-NIPS-[ATOMO: Communication-efficient Learning via Atomic Sparsification](http://papers.nips.cc/paper/8191-atomo-communication-efficient-learning-via-atomic-sparsification)
- 2018-NIPS-[Norm matters: efficient and accurate normalization schemes in deep networks](http://papers.nips.cc/paper/7485-norm-matters-efficient-and-accurate-normalization-schemes-in-deep-networks)
- 2018-NIPS-[Sparsified SGD with memory](http://papers.nips.cc/paper/7697-sparsified-sgd-with-memory)
- 2018-NIPS-[Pelee: A Real-Time Object Detection System on Mobile Devices](http://papers.nips.cc/paper/7466-pelee-a-real-time-object-detection-system-on-mobile-devices)
- 2018-NIPS-[Scalable methods for 8-bit training of neural networks](http://papers.nips.cc/paper/7761-scalable-methods-for-8-bit-training-of-neural-networks)
- 2018-NIPS-[TETRIS: TilE-matching the TRemendous Irregular Sparsity](http://papers.nips.cc/paper/7666-tetris-tile-matching-the-tremendous-irregular-sparsity)
- 2018-NIPS-[Training deep neural networks with 8-bit floating point numbers](http://papers.nips.cc/paper/7994-training-deep-neural-networks-with-8-bit-floating-point-numbers)
- 2018-NIPS-[Multiple instance learning for efficient sequential data classification on resource-constrained devices](http://papers.nips.cc/paper/8292-multiple-instance-learning-for-efficient-sequential-data-classification-on-resource-constrained-devices)
- 2018-NIPS-[Sparse dnns with improved adversarial robustness](https://papers.nips.cc/paper/2018/hash/4c5bde74a8f110656874902f07378009-Abstract.html)
- 2018-NIPSw-[Pruning neural networks: is it time to nip it in the bud?](https://openreview.net/forum?id=r1lbgwFj5m)
- 2018-NIPSw-[Rethinking the Value of Network Pruning](https://openreview.net/forum?id=r1eLk2mKiX) [[2019 ICLR version](https://openreview.net/forum?id=rJlnB3C5Ym)] [[PyTorch Code](https://github.com/Eric-mingjie/rethinking-network-pruning)]
- 2018-NIPSw-[Structured Pruning for Efficient ConvNets via Incremental Regularization](https://openreview.net/forum?id=S1e_xM7_iQ) [[2019 IJCNN version](https://arxiv.org/abs/1804.09461)] [[Caffe Code](https://github.com/MingSun-Tse/Caffe_IncReg)]
- 2018-NIPSw-[Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling](https://openreview.net/forum?id=B1eHgu-Fim)
- 2018-NIPSw-[Learning Sparse Networks Using Targeted Dropout](https://arxiv.org/abs/1905.13678) [[OpenReview](https://openreview.net/forum?id=HkghWScuoQ)] [[Code](https://github.com/for-ai/TD)]
- 2018-WACV-[Recovering from Random Pruning: On the Plasticity of Deep Convolutional Neural Networks](https://arxiv.org/abs/1801.10447)
- 2018.05-[Compression of Deep Convolutional Neural Networks under Joint Sparsity Constraints](https://arxiv.org/abs/1805.08303)
- 2018.05-[AutoPruner: An End-to-End Trainable Filter Pruning Method for Efficient Deep Model Inference](https://arxiv.org/abs/1805.08941)
- 2018.10-[A Closer Look at Structured Pruning for Neural Network Compression](https://arxiv.org/abs/1810.04622) [[Code](https://github.com/BayesWatch/pytorch-prunes)]
- 2018.11-[Second-order Optimization Method for Large Mini-batch: Training ResNet-50 on ImageNet in 35 Epochs](https://arxiv.org/abs/1811.12019)
- 2018.11-[PydMobileNet: Improved Version of MobileNets with Pyramid Depthwise Separable Convolution](https://arxiv.org/abs/1811.07083)

**2019**
- 2019-MLSys-[Towards Federated Learning at Scale: System Design](https://arxiv.org/pdf/1902.01046.pdf)
- 2019-MLsys-[To compress or not to compress: Understanding the Interactions between Adversarial Attacks and Neural Network Compression
](https://arxiv.org/abs/1810.00208)
- 2019-ICLR-[Slimmable Neural Networks](https://openreview.net/forum?id=H1gMCsAqY7) [[Code](https://github.com/JiahuiYu/slimmable_networks)]
- 2019-ICLR-[Defensive Quantization: When Efficiency Meets Robustness](https://arxiv.org/abs/1904.08444)
- 2019-ICLR-[Minimal Random Code Learning: Getting Bits Back from Compressed Model Parameters](https://openreview.net/forum?id=r1f0YiCctm) [[Code](https://github.com/cambridge-mlg/miracle)]
- 2019-ICLR-[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332) [[Code](https://github.com/MIT-HAN-LAB/ProxylessNAS)]
- 2019-ICLR-[SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX)
- 2019-ICLR-[Non-vacuous Generalization Bounds at the ImageNet Scale: a PAC-Bayesian Compression Approach](https://openreview.net/forum?id=BJgqqsAct7)
- 2019-ICLR-[Dynamic Channel Pruning: Feature Boosting and Suppression](https://openreview.net/forum?id=BJxh2j0qYm)
- 2019-ICLR-[Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://openreview.net/forum?id=BylBr3C9K7)
- 2019-ICLR-[RotDCF: Decomposition of Convolutional Filters for Rotation-Equivariant Deep Networks](https://openreview.net/forum?id=H1gTEj09FX)
- 2019-ICLR-[Dynamic Sparse Graph for Efficient Deep Learning](https://openreview.net/forum?id=H1goBoR9F7)
- 2019-ICLR-[Big-Little Net: An Efficient Multi-Scale Feature Representation for Visual and Speech Recognition](https://openreview.net/forum?id=HJMHpjC9Ym)
- 2019-ICLR-[Data-Dependent Coresets for Compressing Neural Networks with Applications to Generalization Bounds](https://openreview.net/forum?id=HJfwJ2A5KX)
- 2019-ICLR-[Learning Recurrent Binary/Ternary Weights](https://openreview.net/forum?id=HkNGYjR9FX)
- 2019-ICLR-[Double Viterbi: Weight Encoding for High Compression Ratio and Fast On-Chip Reconstruction for Deep Neural Network](https://openreview.net/forum?id=HkfYOoCcYX)
- 2019-ICLR-[Relaxed Quantization for Discretized Neural Networks](https://openreview.net/forum?id=HkxjYoCqKX)
- 2019-ICLR-[Integer Networks for Data Compression with Latent-Variable Models](https://openreview.net/forum?id=S1zz2i0cY7)
- 2019-ICLR-[Minimal Random Code Learning: Getting Bits Back from Compressed Model Parameters](https://openreview.net/forum?id=r1f0YiCctm)
- 2019-ICLR-[Analysis of Quantized Models](https://openreview.net/forum?id=ryM_IoAqYX)
- 2019-ICLR-[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) [[Code](https://github.com/quark0/darts)]
- 2019-ICLR-[Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/abs/1810.05749)
- 2019-ICLR-[Learnable Embedding Space for Efficient Neural Architecture Compression](https://openreview.net/forum?id=S1xLN3C9YX) [[Code](https://github.com/Friedrich1006/ESNAC)]
- 2019-ICLR-[Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081)
- 2019-ICLR-[SNAS: stochastic neural architecture search](https://openreview.net/pdf?id=rylqooRqK7)
- 2019-AAAIo-[A layer decomposition-recomposition framework for neuron pruning towards accurate lightweight networks](https://arxiv.org/abs/1812.06611)
- 2019-AAAI-[Balanced Sparsity for Efficient DNN Inference on GPU](https://arxiv.org/abs/1811.00206) [[Code](https://github.com/Howal/balanced-sparsity)]
- 2019-AAAI-[CircConv: A Structured Convolution with Low Complexity](https://arxiv.org/abs/1902.11268)
- 2019-AAAI-[Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548.pdf)
- 2019-AAAI-[Universal Approximation Property and Equivalence of Stochastic Computing-Based Neural Networks and Binary Neural Networks](https://www.aaai.org/ojs/index.php/AAAI/article/view/4475)
- 2019-WACV-[DAC: Data-free Automatic Acceleration of Convolutional Networks](https://arxiv.org/abs/1812.08374)
- 2019-ASPLOS-[Packing Sparse Convolutional Neural Networks for Efficient Systolic Array Implementations: Column Combining Under Joint Optimization](https://arxiv.org/abs/1811.04770)
- 2019-CVPRo-[HAQ: hardware-aware automated quantization](https://arxiv.org/pdf/1811.08886.pdf)
- 2019-CVPRo-[Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250) [[Code](https://github.com/he-y/filter-pruning-geometric-median)]
- 2019-CVPR-[All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification](https://arxiv.org/abs/1903.05285)
- 2019-CVPR-[Importance Estimation for Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html) [[Code](https://github.com/NVlabs/Taylor_pruning)]
- 2019-CVPR-[HetConv Heterogeneous Kernel-Based Convolutions for Deep CNNs](https://arxiv.org/abs/1903.04120)
- 2019-CVPR-[Fully Learnable Group Convolution for Acceleration of Deep Neural Networks](https://arxiv.org/abs/1904.00346)
- 2019-CVPR-[Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291)
- 2019-CVPR-[ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_ChamNet_Towards_Efficient_Network_Design_Through_Platform-Aware_Model_Adaptation_CVPR_2019_paper.pdf)
- 2019-CVPR-[Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/pdf/1903.03777.pdf) [[Code](https://github.com/lixincn2015/Partial-Order-Pruning)]
- 2019-CVPR-[Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf) [[Code](https://github.com/tensorflow/models/tree/master/research/deeplab)]
- 2019-CVPR-[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) [[Code](https://github.com/AnjieZheng/MnasNet-PyTorch)]
- 2019-CVPR-[MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/pdf/1903.06496.pdf)
- 2019-CVPR-[A Neurobiological Evaluation Metric for Neural Network Model Search](https://arxiv.org/pdf/1805.10726.pdf)
- 2019-CVPR-[Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells](https://arxiv.org/abs/1810.10804)
- 2019-CVPR-[Efficient Neural Network Compression](https://arxiv.org/abs/1811.12781) [[Code](https://github.com/Hyeji-Kim/ENC)]
- 2019-CVPR-[T-Net: Parametrizing Fully Convolutional Nets with a Single High-Order Tensor](http://openaccess.thecvf.com/content_CVPR_2019/html/Kossaifi_T-Net_Parametrizing_Fully_Convolutional_Nets_With_a_Single_High-Order_Tensor_CVPR_2019_paper.html)
- 2019-CVPR-[Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837) [[Code](https://github.com/ShawnDing1994/Centripetal-SGD)]
- 2019-CVPR-[DSC: Dense-Sparse Convolution for Vectorized Inference of Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPRW_2019/html/SAIAD/Frickenstein_DSC_Dense-Sparse_Convolution_for_Vectorized_Inference_of_Convolutional_Neural_Networks_CVPRW_2019_paper.html)
- 2019-CVPR-[DupNet: Towards Very Tiny Quantized CNN With Improved Accuracy for Face Detection](http://openaccess.thecvf.com/content_CVPRW_2019/html/EVW/Gao_DupNet_Towards_Very_Tiny_Quantized_CNN_With_Improved_Accuracy_for_CVPRW_2019_paper.html)
- 2019-CVPR-[ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_ECC_Platform-Independent_Energy-Constrained_Deep_Neural_Network_Compression_via_a_Bilinear_CVPR_2019_paper.html)
- 2019-CVPR-[Variational Convolutional Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html)
- 2019-CVPR-[Accelerating Convolutional Neural Networks via Activation Map Compression](http://openaccess.thecvf.com/content_CVPR_2019/html/Georgiadis_Accelerating_Convolutional_Neural_Networks_via_Activation_Map_Compression_CVPR_2019_paper.html)
- 2019-CVPR-[Compressing Convolutional Neural Networks via Factorized Convolutional Filters](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.html)
- 2019-CVPR-[Deep Virtual Networks for Memory Efficient Inference of Multiple Tasks](http://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Deep_Virtual_Networks_for_Memory_Efficient_Inference_of_Multiple_Tasks_CVPR_2019_paper.html)
- 2019-CVPR-[Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Exploiting_Kernel_Sparsity_and_Entropy_for_Interpretable_CNN_Compression_CVPR_2019_paper.html)
- 2019-CVPR-[MBS: Macroblock Scaling for CNN Model Reduction](http://openaccess.thecvf.com/content_CVPR_2019/html/Lin_MBS_Macroblock_Scaling_for_CNN_Model_Reduction_CVPR_2019_paper.html)
- 2019-CVPR-[On Implicit Filter Level Sparsity in Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPR_2019/html/Mehta_On_Implicit_Filter_Level_Sparsity_in_Convolutional_Neural_Networks_CVPR_2019_paper.html)
- 2019-CVPR-[Structured Pruning of Neural Networks With Budget-Aware Regularization](http://openaccess.thecvf.com/content_CVPR_2019/html/Lemaire_Structured_Pruning_of_Neural_Networks_With_Budget-Aware_Regularization_CVPR_2019_paper.html)
- 2019-CVPRo-[Neural Rejuvenation: Improving Deep Network Training by Enhancing Computational Resource Utilization](https://arxiv.org/abs/1812.00481) [[Code](https://github.com/joe-siyuan-qiao/NeuralRejuvenation-CVPR19)]
- 2019-ICML-[Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](https://arxiv.org/abs/1905.04748) [[Code](https://github.com/ShawnDing1994/AOFP)]
- 2019-ICML-[EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis](https://arxiv.org/abs/1905.05934) [[PyTorch Code](https://github.com/alecwangcq/EigenDamage-Pytorch)]
- 2019-ICML-[Zero-Shot Knowledge Distillation in Deep Networks](https://arxiv.org/abs/1905.08114) [[Code](https://github.com/vcl-iisc/ZSKD)]
- 2019-ICML-[LegoNet: Efficient Convolutional Neural Networks with Lego Filters](http://proceedings.mlr.press/v97/yang19c.html) [[Code](https://github.com/zhaohui-yang/LegoNet_pytorch)]
- 2019-ICML-[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) [[Code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)]
- 2019-ICML-[Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html)
- 2019-ICML-[Training CNNs with Selective Allocation of Channels](http://proceedings.mlr.press/v97/jeong19c.html)
- 2019-ICML-[NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) [[Code](https://github.com/google-research/nasbench)]
- 2019-ICML-[Learning fast algorithms for linear transforms using butterfly factorizations](https://arxiv.org/abs/1903.05895)
- 2019-ICMLw-[Towards Learning of Filter-Level Heterogeneous Compression of Convolutional Neural Networks](https://arxiv.org/abs/1904.09872) [[Code](https://github.com/yochaiz/Slimmable)] (AutoML workshop)
- 2019-IJCAI-[Play and Prune: Adaptive Filter Pruning for Deep Model Compression](https://arxiv.org/abs/1905.04446)
- 2019-BigComp-[Towards Robust Compressed Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8679132)
- 2019-ICCV-[Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883)
- 2019-ICCV-[Universally Slimmable Networks and Improved Training Techniques](https://arxiv.org/abs/1903.05134)
- 2019-ICCV-[MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) [[Code](https://github.com/liuzechun/MetaPruning)]
- 2019-ICCV-[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) [[Code](https://github.com/chenxin061/pdarts)]
- 2019-ICCV-[Data-Free Quantization through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721)
- 2019-ICCV-[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930)
- 2019-ICCV-[Adversarial Robustness vs. Model Compression, or Both?](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf) [[PyTorch Code](https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM)]
- 2019-NIPS-[Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/abs/1909.12778)
- 2019-NIPS-[Model Compression with Adversarial Robustness: A Unified Optimization Framework](http://papers.nips.cc/paper/8410-model-compression-with-adversarial-robustness-a-unified-optimization-framework)
- 2019-NIPS-[AutoPrune: Automatic Network Pruning by Regularizing Auxiliary Parameters](https://nips.cc/Conferences/2019/Schedule?showEvent=14303)
- 2019-NIPS-[Double Quantization for Communication-Efficient Distributed Optimization](https://nips.cc/Conferences/2019/Schedule?showEvent=13598)
- 2019-NIPS-[Focused Quantization for Sparse CNNs](https://nips.cc/Conferences/2019/Schedule?showEvent=13686)
- 2019-NIPS-[E2-Train: Training State-of-the-art CNNs with Over 80% Energy Savings](http://papers.nips.cc/paper/8757-e2-train-training-state-of-the-art-cnns-with-over-80-less-energy)
- 2019-NIPS-[MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization](https://papers.nips.cc/paper/8647-metaquant-learning-to-quantize-by-learning-to-penetrate-non-differentiable-quantization)
- 2019-NIPS-[Random Projections with Asymmetric Quantization](https://papers.nips.cc/paper/9268-random-projections-with-asymmetric-quantization)
- 2019-NIPS-[Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) [[Code](https://github.com/D-X-Y/TAS)]
- 2019-NIPS-[Point-Voxel CNN for Efficient 3D Deep Learning](http://papers.nips.cc/paper/8382-point-voxel-cnn-for-efficient-3d-deep-learning) [[Code](https://github.com/mit-han-lab/pvcnn)]
- 2019-NIPS-[Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) [[PyTorch Code](https://github.com/youzhonghui/gate-decorator-pruning)]
- 2019-NIPS-[A Mean Field Theory of Quantized Deep Networks: The Quantization-Depth Trade-Off](https://papers.nips.cc/paper/8926-a-mean-field-theory-of-quantized-deep-networks-the-quantization-depth-trade-off)
- 2019-NIPS-[Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification and Local Computations](https://papers.nips.cc/paper/9610-qsparse-local-sgd-distributed-sgd-with-quantization-sparsification-and-local-computations)
- 2019-NIPS-[Post training 4-bit quantization of convolutional networks for rapid-deployment](https://papers.nips.cc/paper/9008-post-training-4-bit-quantization-of-convolutional-networks-for-rapid-deployment)
- 2019-PR-[Filter-in-Filter: Improve CNNs in a Low-cost Way by Sharing Parameters among the Sub-filters of a Filter](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300640)
- 2019-PRL-[BDNN: Binary Convolution Neural Networks for Fast Object Detection](https://www.sciencedirect.com/science/article/abs/pii/S0167865519301096)
- 2019-TNNLS-[Towards Compact ConvNets via Structure-Sparsity Regularized Filter Pruning](https://arxiv.org/abs/1901.07827) [[Code](https://github.com/ShaohuiLin/SSR)]
- 2019.03-[Network Slimming by Slimmable Networks: Towards One-Shot Architecture Search for Channel Numbers](https://arxiv.org/abs/1903.11728) [[Code](https://github.com/JiahuiYu/slimmable_networks)]
- 2019.03-[Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420)
- 2019.04-[Resource Efficient 3D Convolutional Neural Networks](https://arxiv.org/abs/1904.02422)
- 2019.04-[Meta Filter Pruning to Accelerate Deep Convolutional Neural Networks](https://arxiv.org/abs/1904.03961)
- 2019.04-[Knowledge Squeezed Adversarial Network Compression](https://arxiv.org/abs/1904.05100)
- 2019.05-[Dynamic Neural Network Channel Execution for Efficient Training](https://arxiv.org/abs/1905.06435)
- 2019.06-[AutoGrow: Automatic Layer Growing in Deep Convolutional Networks](https://arxiv.org/abs/1906.02909)
- 2019.06-[BasisConv: A method for compressed representation and learning in CNNs](https://arxiv.org/abs/1906.04509)
- 2019.06-[BlockSwap: Fisher-guided Block Substitution for Network Compression](https://arxiv.org/abs/1906.04113)
- 2019.06-[Separable Layers Enable Structured Efficient Linear Substitutions](https://arxiv.org/abs/1906.00859) [[Code](https://github.com/BayesWatch/deficient-efficient)] 
- 2019.06-[Butterfly Transform: An Efficient FFT Based Neural Architecture Design](https://arxiv.org/abs/1906.02256)
- 2019.06-[A Taxonomy of Channel Pruning Signals in CNNs](https://arxiv.org/abs/1906.04675)
- 2019.08-[Adversarial Neural Pruning with Latent Vulnerability Suppression](https://arxiv.org/abs/1908.04355)
- 2019.09-[Training convolutional neural networks with cheap convolutions and online distillation](https://arxiv.org/abs/1909.13063)
- 2019.09-[Pruning from Scratch](https://arxiv.org/abs/1909.12579)
- 2019.11-[Adversarial Interpolation Training: A Simple Approach for Improving Model Robustness](https://openreview.net/forum?id=Syejj0NYvr)
- 2019.11-[A Programmable Approach to Model Compression](https://arxiv.org/abs/1911.02497) [[Code](https://github.com/NVlabs/condensa)]

**2020**
- 2020-AAAI-[Pconv: The missing but desirable sparsity in dnn weight pruning for real-time execution on mobile devices](https://arxiv.org/abs/1909.05073)
- 2020-AAAI-[Channel Pruning Guided by Classification Loss and Feature Importance](https://arxiv.org/abs/2003.06757)
- 2020-AAAI-[Pruning from Scratch](https://arxiv.org/abs/1909.12579?context=cs.CV)
- 2020-AAAI-[Harmonious Coexistence of Structured Weight Pruning and Ternarization for Deep Neural Networks](https://aaai.org/Papers/AAAI/2020GB/AAAI-YangL.9289.pdf)
- 2020-AAAI-[AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://arxiv.org/abs/1907.03141)
- 2020-AAAI-[DARB: A Density-Adaptive Regular-Block Pruning for Deep Neural Networks](https://arxiv.org/abs/1911.08020)
- 2020-AAAI-[Real-Time Object Tracking via Meta-Learning: Efficient Model Adaptation and One-Shot Channel Pruning](https://arxiv.org/abs/1911.11170)
- 2020-AAAI-[Dynamic Network Pruning with Interpretable Layerwise Channel Selection](https://aaai.org/ojs/index.php/AAAI/article/view/6098)
- 2020-AAAI-[Reborn Filters: Pruning Convolutional Neural Networks with Limited Data](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-TangY.1279.pdf)
- 2020-AAAI-[Layerwise Sparse Coding for Pruned Deep Neural Networks with Extreme Compression Ratio](https://aaai.org/ojs/index.php/AAAI/article/view/5927)
- 2020-AAAI-[Sparsity-inducing Binarized Neural Networks](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-WangP.1440.pdf)
- 2020-AAAI-[Structured Sparsification of Gated Recurrent Neural Networks](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LobachevaE.8844.pdf)
- 2020-AAAI-[Hierarchical Knowledge Squeezed Adversarial Network Compression](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LiP.697.pdf)
- 2020-AAAI-[Embedding Compression with Isotropic Iterative Quantization](https://arxiv.org/abs/2001.05314)
- 2020-ICLR-[Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://openreview.net/forum?id=S1gSj0NKvB) [[Code](https://github.com/lottery-ticket/rewinding-iclr20-public)]
- 2020-ICLR-[Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://openreview.net/forum?id=ryl3ygHYDB) [[Code](https://github.com/alinlab/lookahead_pruning)]
- 2020-ICLR-[Dynamic Model Pruning with Feedback](https://openreview.net/pdf?id=SJem8lSFwB)
- 2020-ICLR-[Provable Filter Pruning for Efficient Neural Networks](https://openreview.net/forum?id=BJxkOlSYDH)
- 2020-ICLR-[Data-Independent Neural Pruning via Coresets](https://openreview.net/forum?id=H1gmHaEKwB)
- 2020-ICLR-[FSNet: Compression of Deep Convolutional Neural Networks by Filter Summary](https://openreview.net/forum?id=S1xtORNFwH)
- 2020-ICLR-[Probabilistic Connection Importance Inference and Lossless Compression of Deep Neural Networks](https://openreview.net/forum?id=HJgCF0VFwr)
- 2020-ICLR-[Neural Epitome Search for Architecture-Agnostic Network Compression](https://openreview.net/forum?id=HyxjOyrKvr)
- 2020-ICLR-[One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation](https://openreview.net/forum?id=r1e9GCNKvH)
- 2020-ICLR-[DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant Sparsity Measures](https://openreview.net/forum?id=rylBK34FDS) [[Code](https://github.com/yanghr/DeepHoyer)]
- 2020-ICLR-[Dynamic Sparse Training: Find Efficient Sparse Network From Scratch With Trainable Masked Layers](https://openreview.net/forum?id=SJlbGJrtDB)
- 2020-ICLR-[Scalable Model Compression by Entropy Penalized Reparameterization](https://openreview.net/forum?id=HkgxW0EYDS)
- 2020-ICLR-[A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307)
- 2020-CVPR-[GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907) [[Code](https://github.com/huawei-noah/ghostnet)]
- 2020-CVPR-[Filter Grafting for Deep Neural Networks](https://arxiv.org/pdf/2001.05868.pdf)
- 2020-CVPR-[Low-rank Compression of Neural Nets: Learning the Rank of Each Layer](http://graduatestudent.ucmerced.edu/yidelbayev/papers/cvpr20/cvpr20a.pdf)
- 2020-CVPR-[Structured Compression by Weight Encryption for Unstructured Pruning and Quantization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kwon_Structured_Compression_by_Weight_Encryption_for_Unstructured_Pruning_and_Quantization_CVPR_2020_paper.pdf)
- 2020-CVPR-[Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.pdf)
- 2020-CVPR-[APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.pdf)
- 2020-CVPR-[Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.pdf) [[Code](https://github.com/ofsoundof/group_sparsity)]
- 2020-CVPR-[Neural Network Pruning With Residual-Connections and Limited-Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Neural_Network_Pruning_With_Residual-Connections_and_Limited-Data_CVPR_2020_paper.pdf)
- 2020-CVPR-[Multi-Dimensional Pruning: A Unified Framework for Model Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.pdf)
- 2020-CVPR-[Discrete Model Compression With Resource Constraint for Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.pdf)
- 2020-CVPR-[Automatic Neural Network Compression by Sparsity-Quantization Joint Learning: A Constrained Optimization-Based Approach](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Automatic_Neural_Network_Compression_by_Sparsity-Quantization_Joint_Learning_A_Constrained_CVPR_2020_paper.pdf)
- 2020-CVPR-[Low-Rank Compression of Neural Nets: Learning the Rank of Each Layer](https://openaccess.thecvf.com/content_CVPR_2020/papers/Idelbayev_Low-Rank_Compression_of_Neural_Nets_Learning_the_Rank_of_Each_CVPR_2020_paper.pdf)
- 2020-CVPR-[The Knowledge Within: Methods for Data-Free Model Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Haroush_The_Knowledge_Within_Methods_for_Data-Free_Model_Compression_CVPR_2020_paper.pdf)
- 2020-CVPR-[GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_GAN_Compression_Efficient_Architectures_for_Interactive_Conditional_GANs_CVPR_2020_paper.pdf) [[Code](https://github.com/mit-han-lab/gan-compression)]
- 2020-CVPR-[Few Sample Knowledge Distillation for Efficient Network Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Few_Sample_Knowledge_Distillation_for_Efficient_Network_Compression_CVPR_2020_paper.pdf)
- 2020-CVPR-[Fast sparse convnets](https://openaccess.thecvf.com/content_CVPR_2020/html/Elsen_Fast_Sparse_ConvNets_CVPR_2020_paper.html)
- 2020-CVPR-[Structured Multi-Hashing for Model Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Eban_Structured_Multi-Hashing_for_Model_Compression_CVPR_2020_paper.pdf)
- 2020-CVPRo-[AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200) [[Code](https://github.com/huawei-noah/AdderNet)]
- 2020-CVPRo-[Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368) [[Code](https://github.com/cmu-enyac/LeGR)]
- 2020-CVPRo-[HRank: Filter Pruning Using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf) [[Code](https://github.com/lmbxmu/HRank)]
- 2020-CVPRo-[DaST: Data-free Substitute Training for Adversarial Attacks](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_DaST_Data-Free_Substitute_Training_for_Adversarial_Attacks_CVPR_2020_paper.html) [[Code](https://github.com/zhoumingyi/DaST)]
- 2020-ICML-[PENNI: Pruned Kernel Sharing for Efficient CNN Inference](https://arxiv.org/abs/2005.07133) [[Code](https://github.com/timlee0212/PENNI)]
- 2020-ICML-[Operation-Aware Soft Channel Pruning using Differentiable Masks](https://arxiv.org/abs/2007.03938)
- 2020-ICML-[DropNet: Reducing Neural Network Complexity via Iterative Pruning](https://proceedings.icml.cc/static/paper_files/icml/2020/2026-Paper.pdf)
- 2020-ICML-[Network Pruning by Greedy Subnetwork Selection](https://arxiv.org/abs/2003.01794)
- 2020-ICML-[AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks](https://arxiv.org/abs/2006.08198)
- 2020-ICML-[Soft Threshold Weight Reparameterization for Learnable Sparsity](https://arxiv.org/abs/2002.03231) [[PyTorch Code](https://github.com/RAIVNLab/STR)]
- 2020-ICML-[Activation sparsity: Inducing and exploiting activation sparsity for fast inference on deep neural networks](http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf)
- 2020-EMNLP-[Structured Pruning of Large Language Models](https://arxiv.org/abs/1910.04732) [[Code](https://github.com/asappresearch/flop)]
- 2020-NIPS-[Pruning neural networks without any data by iteratively conserving synaptic flow](https://papers.nips.cc/paper/2020/hash/46a4378f835dc8040c8057beb6a2da52-Abstract.html) 
- 2020-NIPS-[Neuron-level Structured Pruning using Polarization Regularizer](https://papers.nips.cc/paper/2020/hash/703957b6dd9e3a7980e040bee50ded65-Abstract.html)
- 2020-NIPS-[SCOP: Scientific Control for Reliable Neural Network Pruning](https://papers.nips.cc/paper/2020/hash/7bcdf75ad237b8e02e301f4091fb6bc8-Abstract.html)
- 2020-NIPS-[Directional Pruning of Deep Neural Networks](https://papers.nips.cc/paper/2020/hash/a09e75c5c86a7bf6582d2b4d75aad615-Abstract.html)
- 2020-NIPS-[Storage Efficient and Dynamic Flexible Runtime Channel Pruning via Deep Reinforcement Learning](https://papers.nips.cc/paper/2020/hash/a914ecef9c12ffdb9bede64bb703d877-Abstract.html)
- 2020-NIPS-[Pruning Filter in Filter](https://papers.nips.cc/paper/2020/hash/ccb1d45fb76f7c5a0bf619f979c6cf36-Abstract.html)
- 2020-NIPS-[HYDRA: Pruning Adversarially Robust Neural Networks](https://papers.nips.cc/paper/2020/hash/e3a72c791a69f87b05ea7742e04430ed-Abstract.html)
- 2020-NIPS-[Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://papers.nips.cc/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html)
- 2020-NIPS-[Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://papers.nips.cc/paper/2020/hash/eae27d77ca20db309e056e3d2dcd7d69-Abstract.html)
- 2020-NIPS-[Position-based Scaled Gradient for Model Quantization and Pruning](https://papers.nips.cc/paper/2020/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)
- 2020-NIPS-[The Generalization-Stability Tradeoff In Neural Network Pruning](https://papers.nips.cc/paper/2020/hash/ef2ee09ea9551de88bc11fd7eeea93b0-Abstract.html)
- 2020-NIPS-[FleXOR: Trainable Fractional Quantization](https://papers.nips.cc/paper/2020/hash/0e230b1a582d76526b7ad7fc62ae937d-Abstract.html)
- 2020-NIPS-[Adaptive Gradient Quantization for Data-Parallel SGD](https://papers.nips.cc/paper/2020/hash/20b5e1cf8694af7a3c1ba4a87f073021-Abstract.html)
- 2020-NIPS-[Robust Quantization: One Model to Rule Them All](https://papers.nips.cc/paper/2020/hash/3948ead63a9f2944218de038d8934305-Abstract.html)
- 2020-NIPS-[HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks](https://papers.nips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html)
- 2020-NIPS-[Efficient Exact Verification of Binarized Neural Networks](https://papers.nips.cc/paper/2020/hash/1385974ed5904a438616ff7bdb3f7439-Abstract.html)
- 2020-NIPS-[Ultra-Low Precision 4-bit Training of Deep Neural Networks](https://papers.nips.cc/paper/2020/hash/13b919438259814cd5be8cb45877d577-Abstract.html)
- 2020-NIPS-[Path Sample-Analytic Gradient Estimators for Stochastic Binary Networks](https://papers.nips.cc/paper/2020/hash/96fca94df72984fc97ee5095410d4dec-Abstract.html)
- 2020-NIPS-[Fast fourier convolution](https://papers.nips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html)

**2021**
- 2021-WACV-[CAP: Context-Aware Pruning for Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2021/papers/He_CAP_Context-Aware_Pruning_for_Semantic_Segmentation_WACV_2021_paper.pdf) [[Code](https://github.com/erichhhhho/CAP-Context-Aware-Pruning-for-Semantic-Segmentation)]
- 2021-AAAI-[Few Shot Network Compression via Cross Distillation](https://arxiv.org/abs/1911.09450)
- 2021-AAAI-[Conditional Channel Pruning for Automated Model Compression]() [[Code](https://github.com/liuyixin-louis/CAMC-hanlab)]
- 2021-ICLR-[Neural Pruning via Growing Regularization](https://openreview.net/forum?id=o966_Is_nPA) [[PyTorch Code](https://github.com/MingSun-Tse/Regularization-Pruning)]
- 2021-ICLR-[Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP)
- 2021-ICLR-[ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations](https://openreview.net/forum?id=xCxXwTzx4L1)
- 2021-ICLR-[A Gradient Flow Framework For Analyzing Network Pruning](https://openreview.net/forum?id=rumv7QmLUue) (Spotlight)
- 2021-CVPR-[Towards Compact CNNs via Collaborative Compression](https://arxiv.org/abs/2105.11228)
- 2021-CVPR-[Manifold Regularized Dynamic Network Pruning](https://arxiv.org/abs/2103.05861)
- 2021-CVPR-[Learnable Companding Quantization for Accurate Low-bit Neural Networks](https://arxiv.org/abs/2103.07156)
- 2021-CVPR-[Diversifying Sample Generation for Accurate Data-Free Quantization](https://arxiv.org/abs/2103.01049)
- 2021-CVPR-[Zero-shot Adversarial Quantization](https://arxiv.org/abs/2103.15263) [Oral] [[Code](https://github.com/FLHonker/ZAQ-code)]
- 2021-CVPR-[Network Quantization with Element-wise Gradient Scaling](https://arxiv.org/abs/2104.00903) [[Project](https://cvlab.yonsei.ac.kr/projects/EWGS/)]
- 2021-ICML-[Group Fisher Pruning for Practical Network Compression](http://proceedings.mlr.press/v139/liu21ab.html) [[Code](https://github.com/jshilong/FisherPruning)]
- 2021-ICML-[Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](http://proceedings.mlr.press/v139/wang21e.html) 
- 2021-ICML-[A Probabilistic Approach to Neural Network Pruning](https://arxiv.org/abs/2105.10065)
- 2021-ICML-[On the Predictability of Pruning Across Scales](http://proceedings.mlr.press/v139/rosenfeld21a.html)
- 2021-ICML-[Sparsifying Networks via Subdifferential Inclusion](http://proceedings.mlr.press/v139/verma21b.html)
- 2021-ICML-[Selfish Sparse RNN Training](https://arxiv.org/abs/2101.09048) [[Code](https://github.com/Shiweiliuiiiiiii/Selfish-RNN)]
- 2021-ICML-[Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](https://arxiv.org/abs/2102.02887) [[Code](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization)]
- 2021-ICML-[Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling](http://proceedings.mlr.press/v139/ozdenizci21a.html)
- 2021-ICML-[ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/abs/2104.14129)
- 2021-ICML-[Leveraging Sparse Linear Layers for Debuggable Deep Networks](https://arxiv.org/abs/2105.04857)
- 2021-ICML-[PHEW: Constructing Sparse Networks that Learn Fast and Generalize Well without Training Data](http://proceedings.mlr.press/v139/patil21a.html)
- 2021-ICML-[BASE Layers: Simplifying Training of Large, Sparse Models](https://arxiv.org/abs/2103.16716) [[Code](https://github.com/pytorch/fairseq/)]
- 2021-ICML-[Dense for the Price of Sparse: Improved Performance of Sparsely Initialized Networks via a Subspace Offset](https://arxiv.org/abs/2102.07655)
- 2021-ICML-[I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321)
- 2021-ICML-[Training Quantized Neural Networks to Global Optimality via Semidefinite Programming](https://arxiv.org/abs/2105.01420)
- 2021-ICML-[Differentiable Dynamic Quantization with Mixed Precision and Adaptive Resolution](https://arxiv.org/abs/2106.02295)
- 2021-ICML-[Communication-Efficient Distributed Optimization with Quantized Preconditioners](https://arxiv.org/abs/2102.07214)
- 2021-NIPS-[Aligned Structured Sparsity Learning for Efficient Image Super-Resolution](https://papers.nips.cc/paper/2021/file/15de21c670ae7c3f6f3f1f37029303c9-Paper.pdf) [[Code](https://github.com/MingSun-Tse/ASSL)] (Spotlight!)
- 2021-NIPS-[Scatterbrain: Unifying Sparse and Low-rank Attention](https://arxiv.org/abs/2110.15343) [[Code](https://github.com/HazyResearch/scatterbrain)]
- 2021-NIPS-[Only Train Once: A One-Shot Neural Network Training And Pruning Framework](https://proceedings.neurips.cc/paper/2021/file/a376033f78e144f494bfc743c0be3330-Paper.pdf) [[Code](https://github.com/tianyic/only_train_once)]
- 2021-NIPS-[CHIP: CHannel Independence-based Pruning for Compact Neural Networks](http://128.84.21.203/pdf/2110.13981) [[Code](https://github.com/Eclipsess/CHIP_NeurIPS2021)]
- 2021.5-[Dynamical Isometry: The Missing Ingredient for Neural Network Pruning](https://arxiv.org/abs/2105.05916)

#### 2022
- 2022-AAAI-[Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better](https://arxiv.org/abs/2112.09824) [[Code](https://github.com/bibikar/feddst)]
- 2022-ICLR-[Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models](https://arxiv.org/abs/2112.00029)
- 2022-NIPS-[Pruning has a disparate impact on model accuracy](https://openreview.net/forum?id=11nMVZK0WYM)
- 2022-NIPS-[Pruning Neural Networks via Coresets and Convex Geometry: Towards No Assumptions](https://openreview.net/forum?id=btpIaJiRx6z)
- 2022-NIPS-[Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://openreview.net/forum?id=ksVGCOlOEba)
- 2022-NIPS-[Data-Efficient Structured Pruning via Submodular Optimization](https://openreview.net/forum?id=K2QGzyLwpYG)
- 2022-NIPS-[A Fast Post-Training Pruning Framework for Transformers](https://openreview.net/forum?id=0GRBKLBjJE)
- 2022-NIPS-[SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://openreview.net/forum?id=w5DacXWzQ-Q)
- 2022-NIPS-[Recall Distortion in Neural Network Pruning and the Undecayed Pruning Algorithm](https://openreview.net/forum?id=5hgYi4r5MDp)
- 2022-NIPS-[Structural Pruning via Latency-Saliency Knapsack](https://openreview.net/forum?id=cUOR-_VsavA)
- 2022-NIPS-[Sparse Probabilistic Circuits via Pruning and Growing](https://openreview.net/forum?id=KieCChVB6mN)
- 2022-NIPS-[Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropogation](https://openreview.net/forum?id=mTXQIpXPDbh)
- 2022-NIPS-[SInGE: Sparsity via Integrated Gradients Estimation of Neuron Relevance](https://openreview.net/forum?id=oQIJsMlyaW_)
- 2022-NIPS-[VTC-LFC: Vision Transformer Compression with Low-Frequency Components](https://openreview.net/forum?id=HuiLIB6EaOk)
- 2022-NIPS-[Weighted Mutual Learning with Diversity-Driven Model Compression](https://openreview.net/forum?id=UQJoGBNRX4)
- 2022-NIPS-[Resource-Adaptive Federated Learning with All-In-One Neural Composition](https://openreview.net/forum?id=wfel7CjOYk)
- 2022-NIPS-[Controlled Sparsity via Constrained Optimization or: How I Learned to Stop Tuning Penalties and Love Constraints](https://openreview.net/forum?id=XUvSYc6TqDF)
- 2022-NIPS-[On Measuring Excess Capacity in Neural Networks](https://openreview.net/forum?id=l2CVt1ySC2Q)
- 2022-NIPS-[Prune and distill: similar reformatting of image information along rat visual cortex and deep neural networks](https://openreview.net/forum?id=2OpRgzLhoPQ)
- 2022-NIPS-[Deep Compression of Pre-trained Transformer Models](https://openreview.net/forum?id=EZQnauHn-77)
- 2022-NIPS-[Sparsity in Continuous-Depth Neural Networks](https://openreview.net/forum?id=HZ20IYYAwah)
- 2022-NIPS-[Spartan: Differentiable Sparsity via Regularized Transportation](https://openreview.net/forum?id=u4KagP_FjB)
- 2022-NIPS-[Accelerated Projected Gradient Algorithms for Sparsity Constrained Optimization Problems](https://openreview.net/forum?id=0Z0xltoU1q)
- 2022-NIPS-[Feature Learning in L2-regularized DNNs: Attraction/Repulsion and Sparsity](https://openreview.net/forum?id=kK200QKfvjB)
- 2022-NIPS-[Learning Best Combination for Efficient N:M Sparsity](https://openreview.net/forum?id=tbdk6XLYmZj)
- 2022-NIPS-[Accelerating Sparse Convolution with Column Vector-Wise Sparsity](https://openreview.net/forum?id=Q5kXC6hCr1)
- 2022-NIPS-[Differentially Private Model Compression](https://openreview.net/forum?id=68EuccCtO5i)
- 2022-NIPS-[Transformers meet Stochastic Block Models: Attention with Data-Adaptive Sparsity and Cost](https://openreview.net/forum?id=w_jvWzNXd6n)
- 2022-NIPS-[A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models](https://openreview.net/forum?id=UmaiVbwN1v)
- 2022-NIPS-[Make Sharpness-Aware Minimization Stronger: A Sparsified Perturbation Approach](https://openreview.net/forum?id=88_wNI6ZBDZ)
- 2022-NIPS-[Learning sparse features can lead to overfitting in neural networks](https://openreview.net/forum?id=dZEZu7zxJBF)
- 2022-NIPS-[Deep Architecture Connectivity Matters for Its Convergence: A Fine-Grained Analysis](https://openreview.net/forum?id=edgCBcwZxgd)
- 2022-NIPS-[EfficientFormer: Vision Transformers at MobileNet Speed](https://openreview.net/forum?id=NXHXoYMLIG)
- 2022-NIPS-[Revisiting Sparse Convolutional Model for Visual Recognition](https://openreview.net/forum?id=INzRLBAA4JX)
- 2022-NIPS-[Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://openreview.net/forum?id=AUz5Oig77OS)
- 2022-NIPS-[A Theoretical View on Sparsely Activated Networks](https://openreview.net/forum?id=AODVskSug8)
- 2022-NIPS-[Dynamic Sparse Network for Time Series Classification: Learning What to “See”](https://openreview.net/forum?id=ZxOO5jfqSYw)
- 2022-NIPS-[Spatial Pruned Sparse Convolution for Efficient 3D Object Detection](https://openreview.net/forum?id=QqWqFLbllZh)
- 2022-NIPS-[Sparse Structure Search for Delta Tuning](https://openreview.net/forum?id=oOte_397Q4P)
- 2022-NIPS-[Beyond L1: Faster and Better Sparse Models with skglm](https://openreview.net/forum?id=n0dD3d54Wgf)
- 2022-NIPS-[On the Representation Collapse of Sparse Mixture of Experts](https://openreview.net/forum?id=mWaYC6CZf5)
- 2022-NIPS-[M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design](https://openreview.net/forum?id=cFOhdl1cyU-)
- 2022-NIPS-[On-Device Training Under 256KB Memory](https://openreview.net/forum?id=zGvRdBW06F5)
- 2022-NIPS-[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://openreview.net/forum?id=rBCvMG-JsPd)

#### 2023
- 2023-ICLR-[Trainability Preserving Neural Pruning](https://arxiv.org/abs/2207.12534) [[Code](https://github.com/MingSun-Tse/TPP)]
- 2023-ICLR-[NTK-SAP: Improving neural network pruning by aligning training dynamics](https://openreview.net/forum?id=-5EWhW_4qWP) [[Code](https://github.com/YiteWang/NTK-SAP)]
- 2023-CVPR-[DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900)[[code](https://github.com/VainF/Torch-Pruning)]
- 2023-CVPR-[CP3: Channel Pruning Plug-in for Point-based Networks](https://arxiv.org/abs/2303.13097)
- 2023-ICML-[UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers](https://proceedings.mlr.press/v202/shi23e.html) [[code](https://github.com/sdc17/UPop)]
- 2023-ICML-[Gradient-Free Structured Pruning with Unlabeled Data](https://proceedings.mlr.press/v202/nova23a.html)
- 2023-ICML-[Reconstructive Neuron Pruning for Backdoor Defense](https://proceedings.mlr.press/v202/li23v.html) [[code](https://github.com/bboylyg/RNP)]
- 2023-ICML-[UPSCALE: Unconstrained Channel Pruning](https://proceedings.mlr.press/v202/wan23a.html) [[code](https://github.com/apple/ml-upscale)]
- 2023-ICML-[SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html) [[code](https://github.com/IST-DASLab/sparsegpt)]
- 2023-ICML-[Why Random Pruning Is All We Need to Start Sparse](https://proceedings.mlr.press/v202/gadhikar23a.html) [[code](https://github.com/RelationalML/sparse_to_sparse)]
- 2023-ICML-[Fast as CHITA: Neural Network Pruning with Combinatorial Optimization](https://proceedings.mlr.press/v202/benbaki23a.html) [[code](https://github.com/mazumder-lab/CHITA)]
- 2023-ICML-[A Three-regime Model of Network Pruning](https://proceedings.mlr.press/v202/zhou23p.html) [[code](https://github.com/YefanZhou/ThreeRegimePruning)]
- 2023-ICML-[Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://proceedings.mlr.press/v202/jaiswal23b.html) [[code](https://github.com/VITA-Group/instant_soup)]
- 2023-ICML-[Pruning via Sparsity-indexed ODE: a Continuous Sparsity Viewpoint](https://proceedings.mlr.press/v202/mo23c/mo23c.pdf) [[code](https://github.com/mzf666/sparsity-indexed-ode)]

---
### Papers [Actual Acceleration via Sparsity]
- 2018-ICML-[Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435)
- 2018-NIPS-[Tetris: Tile-matching the tremendous irregular sparsity](https://papers.nips.cc/paper/2018/hash/89885ff2c83a10305ee08bd507c1049c-Abstract.html)
- 2021.4-[Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378) (White paper from NVIDIA)
- 2021-ICLR-[Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch](https://openreview.net/forum?id=K9bw7vqp_s) [[Code](https://github.com/NM-sparsity/NM-sparsity)]
- 2021-NIPS-[Channel Permutations for N: M Sparsity](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) [[Code: NVIDIA ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity)]
- 2021-NIPS-[Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks](https://openreview.net/forum?id=vRWZsBLKqA)
- 2021-ICLR-[Learning N:M fine-grained structured sparse neural networks from scratch](https://arxiv.org/abs/2102.04010) [[Code](https://github.com/NM-sparsity/NM-sparsity)] [[Slides](https://iclr.cc/media/iclr-2021/Slides/3174.pdf)]
- 2022-NIPS-[UDC: Unified DNAS for Compressible TinyML Models for Neural Processing Units](https://openreview.net/forum?id=ZJe-XahpyBf)


---
### Papers [Lottery Ticket Hypothesis (LTH)]
For LTH and other _Pruning at Initialization_ papers, please refer to [Awesome-Pruning-at-Initialization](https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization). 


---
### Papers [Bayesian Compression]
- 1995-Neural Computation-[Bayesian Regularisation and Pruning using a Laplace Prior](https://www.researchgate.net/profile/Peter_Williams19/publication/2719575_Bayesian_Regularisation_and_Pruning_using_a_Laplace_Prior/links/58fde123aca2728fa70f6aab/Bayesian-Regularisation-and-Pruning-using-a-Laplace-Prior.pdf)
- 1997-Neural Networks-[Regularization with a Pruning Prior](https://www.sciencedirect.com/science/article/pii/S0893608097000270?casa_token=sLb4dFBnyH8AAAAA:a9WwAAoYl5CgLepZGXjZ5DKQ4YBEjINgGd7Jl2bPHqrbhIWZHso-uC_gpL-85JmdxG7g8x71)
- 2015-NIPS-[Bayesian dark knowledge](http://papers.nips.cc/paper/5965-bayesian-dark-knowledge.pdf)
- 2017-NIPS-[Bayesian Compression for Deep Learning](http://papers.nips.cc/paper/6921-bayesian-compression-for-deep-learning) [[Code](https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL)]
- 2017-ICML-[Variational dropout sparsifies deep neural networks](https://arxiv.org/pdf/1701.05369.pdf)
- 2017-NIPSo-[Structured Bayesian Pruning via Log-Normal Multiplicative Noise](http://papers.nips.cc/paper/7254-structured-bayesian-pruning-via-log-normal-multiplicative-noise)
- 2017-ICMLw-[Bayesian Sparsification of Recurrent Neural Networks](https://arxiv.org/abs/1708.00077)
- 2020-NIPS-[Bayesian Bits: Unifying Quantization and Pruning](https://papers.nips.cc/paper/2020/hash/3f13cf4ddf6fc50c0d39a1d5aeb57dd8-Abstract.html)



## Papers [Knowledge Distillation (KD)]
**Before 2014**
- 1996-[Born again trees](ftp://ftp.stat.berkeley.edu/pub/users/breiman/BAtrees.ps) (proposed compressing neural networks and multipletree predictors by approximating them with a single tree)
- 2006-SIGKDD-[Model compression](https://dl.acm.org/citation.cfm?id=1150464)
- 2010-ML-[A theory of learning from different domains](https://link.springer.com/content/pdf/10.1007%2Fs10994-009-5152-4.pdf)

**2014**
- 2014-NIPS-[Do deep nets really need to be deep?](https://arxiv.org/abs/1312.6184)
- 2014-NIPSw-[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf) [[Code](https://github.com/peterliht/knowledge-distillation-pytorch)]

**2016**
- 2016-ICLR-[Net2net: Accelerating learning via knowledge transfer](https://arxiv.org/abs/1511.05641)
- 2016-ECCV-[Accelerating convolutional neural networks with dominant convolutional kernel and knowledge pre-regression](https://www.researchgate.net/publication/308277663_Accelerating_Convolutional_Neural_Networks_with_Dominant_Convolutional_Kernel_and_Knowledge_Pre-regression)

**2017**
- 2017-ICLR-[Paying more attention to attention: Improving the performance of convolutional neural networksvia attention transfer](http://arxiv.org/abs/1612.03928)
- 2017-ICLR-[Do deep convolutional nets really need to be deep and convolutional?](https://arxiv.org/pdf/1603.05691.pdf)
- 2017-CVPR-[A gift from knowledge distillation: Fast optimization, network minimization and transfer learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)
- 2017-BMVC-[Adapting models to signal degradation using distillation](https://arxiv.org/abs/1604.00433)
- 2017-NIPS-[Sobolev training for neural networks](http://papers.nips.cc/paper/7015-sobolev-training-for-neural-networks.pdf)
- 2017-NIPS-[Learning efficient object detection models with knowledge distillation](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation)
- 2017-NIPSw-[Data-Free Knowledge Distillation for Deep Neural Networks](https://arxiv.org/abs/1710.07535) [[Code](https://github.com/iRapha/replayed_distillation)]
- 2017.07-[Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/pdf/1707.01219.pdf)
- 2017.10-[Knowledge Projection for Deep Neural Networks](https://arxiv.org/abs/1710.09505)
- 2017.11-[Distilling a Neural Network Into a Soft Decision Tree](https://arxiv.org/abs/1711.09784)
- 2017.12-[Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440)

**2018**
- 2018-AAAI-[DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer](https://arxiv.org/abs/1707.01220)
- 2018-AAAI-[Dynamic deep neural networks: Optimizing accuracy-efficiency trade-offs by selective execution](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16291/16575)
- 2018-AAAI-[Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net](https://arxiv.org/abs/1708.04106)
- 2018-AAAI-[Adversarial Learning of Portable Student Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16509)
- 2018-AAAI-[Knowledge Distillation in Generations: More Tolerant Teachers Educate Better Students](https://arxiv.org/abs/1805.05551)
- 2018-ICLR-[Large scale distributed neural network training through online distillation](https://openreview.net/forum?id=rkr1UDeC-)
- 2018-CVPR-[Deep mutual learning](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)
- 2018-ICML-[Born-Again Neural Networks](https://arxiv.org/pdf/1805.04770.pdf)
- 2018-IJCAI-[Better and Faster: Knowledge Transfer from Multiple Self-supervised Learning Tasks via Graph Distillation for Video Classification](https://arxiv.org/abs/1804.10069)
- 2018-ECCV-[2018-ECCV-Learning deep representations with probabilistic knowledge transfer](http://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html) [[Code](https://github.com/passalis/probabilistic_kt)]
- 2018-ECCV-[Graph adaptive knowledge transfer for unsupervised domain adaptation](http://openaccess.thecvf.com/content_ECCV_2018/html/Zhengming_Ding_Graph_Adaptive_Knowledge_ECCV_2018_paper.html)
- 2018-SIGKDD-[Towards Evolutionary Compression](https://www.researchgate.net/profile/Yunhe_Wang3/publication/326502551_Towards_Evolutionary_Compression/links/5b7e9304a6fdcc5f8b5e4fe5/Towards-Evolutionary-Compression.pdf)
- 2018-NIPS-[KDGAN: knowledge distillation with generative adversarial networks](http://papers.nips.cc/paper/7358-kdgan-knowledge-distillation-with-generative-adversarial-networks) [[2019 TPAMI version](https://ieeexplore.ieee.org/abstract/document/8845633)]
- 2018-NIPS-[Knowledge Distillation by On-the-Fly Native Ensemble](http://papers.nips.cc/paper/7980-knowledge-distillation-by-on-the-fly-native-ensemble)
- 2018-NIPS-[Paraphrasing Complex Network: Network Compression via Factor Transfer](http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer)
- 2018-NIPSw-[Variational Mutual Information Distillation for Transfer Learning](http://hushell.github.io/papers/nips18_cl.pdf) [workshop: continual learning](https://sites.google.com/view/continual2018/)
- 2018-NIPSw-[Transparent Model Distillation](https://arxiv.org/pdf/1801.08640.pdf)
- 2018.03-[Interpreting Deep Classifier by Visual Distillation of Dark Knowledge](https://arxiv.org/abs/1803.04042)
- 2018.11-[Dataset Distillation](https://arxiv.org/abs/1811.10959) [[Code](https://github.com/SsnL/dataset-distillation)]
- 2018.12-[Learning Student Networks via Feature Embedding](https://arxiv.org/abs/1812.06597)
- 2018.12-[Few Sample Knowledge Distillation for Efficient Network Compression](https://arxiv.org/abs/1812.01839)

**2019**
- 2019-AAAI-[Knowledge Distillation with Adversarial Samples Supporting Decision Boundary](https://arxiv.org/abs/1805.05532)
- 2019-AAAI-[Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://arxiv.org/abs/1811.03233) [[Code](https://github.com/bhheo/AB_distillation)]
- 2019-AAAI-[Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks](https://arxiv.org/abs/1811.02759) [[Code](https://github.com/cardwing/Codes-for-Steering-Control)]
- 2019-CVPR-[Knowledge Representing: Efficient, Sparse Representation of Prior Knowledge for Knowledge Distillation](http://openaccess.thecvf.com/content_CVPRW_2019/html/CEFRL/Liu_Knowledge_Representing_Efficient_Sparse_Representation_of_Prior_Knowledge_for_Knowledge_CVPRW_2019_paper.html)
- 2019-CVPR-[Knowledge Distillation via Instance Relationship Graph](http://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.html)
- 2019-CVPR-[Variational Information Distillation for Knowledge Transfer](http://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html)
- 2019-CVPR-[Learning Metrics from Teachers Compact Networks for Image Embedding](http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Learning_Metrics_From_Teachers_Compact_Networks_for_Image_Embedding_CVPR_2019_paper.html) [[Code](https://github.com/yulu0724/EmbeddingDistillation)]
- 2019-ICCV-[A Comprehensive Overhaul of Feature Distillation](http://openaccess.thecvf.com/content_ICCV_2019/html/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.html)
- 2019-ICCV-[Similarity-Preserving Knowledge Distillation](https://arxiv.org/abs/1907.09682)
- 2019-ICCV-[Correlation Congruence for Knowledge Distillation](https://arxiv.org/abs/1904.01802)
- 2019-ICCV-[Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)
- 2019-ICCV-[Learning Lightweight Lane Detection CNNs by Self Attention Distillation](https://arxiv.org/abs/1908.00821) [[Code](https://github.com/cardwing/Codes-for-Lane-Detection)]
- 2019-ICCV-[Attention bridging network for knowledge transfer](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Attention_Bridging_Network_for_Knowledge_Transfer_ICCV_2019_paper.html)
- 2019-NIPS-[Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://papers.nips.cc/paper/9151-zero-shot-knowledge-transfer-via-adversarial-belief-matching) [[Code](https://github.com/polo5/ZeroShotKnowledgeTransfer)] (spotlight)
- 2019.05-[DistillHash: Unsupervised Deep Hashing by Distilling Data Pairs](https://arxiv.org/abs/1905.03465)

**2020**
- 2020-ICLR-[Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699) [[Code](https://github.com/HobbitLong/RepDistiller)]
- 2020-AAAI-[A Knowledge Transfer Framework for Differentially Private Sparse Learning]()
- 2020-AAAI-[Uncertainty-aware Multi-shot Knowledge Distillation for Image-based Object Re-identification]()
- 2020-AAAI-[Improved Knowledge Distillation via Teacher Assistant]()
- 2020-AAAI-[Knowledge Distillation from Internal Representations]()
- 2020-AAAI-[Distilling Knowledge from Well-informed Soft Labels for Neural Relation Extraction]()
- 2020-AAAI-[Online Knowledge Distillation with Diverse Peers](https://arxiv.org/abs/1912.00350)
- 2020-AAAI-[Ultrafast Video Attention Prediction with Coupled Knowledge Distillation](https://arxiv.org/abs/1904.04449)
- 2020-AAAI-[Graph Few-shot Learning via Knowledge Transfer]()
- 2020-AAAI-[Diversity Transfer Network for Few-Shot Learning]()
- 2020-AAAI-[Few Shot Network Compression via Cross Distillation](https://arxiv.org/abs/1911.09450)
- 2020-ICLR-[Knowledge Consistency between Neural Networks and Beyond](https://openreview.net/pdf?id=BJeS62EtwH)
- 2020-ICLR-[Contrastive Representation Distillation](https://openreview.net/pdf?id=SkgpBJrtvS) [[Code](http://github.com/HobbitLong/RepDistiller)]
- 2020-ICLR-[BlockSwap: Fisher-guided Block Substitution for Network Compression on a Budget](https://openreview.net/forum?id=SklkDkSFPB)
- 2020-ICLR-[Ensemble Distribution Distillation](https://openreview.net/pdf?id=BygSP6Vtvr)
- 2020-CVPR-[Collaborative Distillation for Ultra-Resolution Universal Style Transfer](https://arxiv.org/abs/2003.08436) [[Code](https://github.com/MingSun-Tse/Collaborative-Distillation)]
- 2020-CVPR-[Explaining Knowledge Distillation by Quantifying the Knowledge](https://arxiv.org/abs/2003.03622)
- 2020-CVPR-[Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252) [[Code](https://github.com/google-research/noisystudent)]
- 2020-CVPR-[Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation From a Blackbox Model](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Neural_Networks_Are_More_Productive_Teachers_Than_Human_Raters_Active_CVPR_2020_paper.pdf)
- 2020-CVPR-[Heterogeneous Knowledge Distillation Using Information Flow Modeling](https://openaccess.thecvf.com/content_CVPR_2020/papers/Passalis_Heterogeneous_Knowledge_Distillation_Using_Information_Flow_Modeling_CVPR_2020_paper.pdf)
- 2020-CVPR-[Creating Something From Nothing: Unsupervised Knowledge Distillation for Cross-Modal Hashing](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Creating_Something_From_Nothing_Unsupervised_Knowledge_Distillation_for_Cross-Modal_Hashing_CVPR_2020_paper.pdf)
- 2020-CVPR-[Revisiting Knowledge Distillation via Label Smoothing Regularization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)
- 2020-CVPR-[Distilling Knowledge From Graph Convolutional Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Distilling_Knowledge_From_Graph_Convolutional_Networks_CVPR_2020_paper.pdf)
- 2020-CVPR-[MineGAN: Effective Knowledge Transfer From GANs to Target Domains With Few Images](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_MineGAN_Effective_Knowledge_Transfer_From_GANs_to_Target_Domains_With_CVPR_2020_paper.pdf) [[Code](https://github.com/yaxingwang/MineGAN)]
- 2020-CVPRo-[Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.pdf) [[Code](https://github.com/NVlabs/DeepInversion)]
- 2020-CVPR-[Online Knowledge Distillation via Collaborative Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Online_Knowledge_Distillation_via_Collaborative_Learning_CVPR_2020_paper.pdf)
- 2020-CVPR-[Distilling Cross-Task Knowledge via Relationship Matching](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Distilling_Cross-Task_Knowledge_via_Relationship_Matching_CVPR_2020_paper.pdf)
- 2020-CVPR-[Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Data-Free_Knowledge_Amalgamation_via_Group-Stack_Dual-GAN_CVPR_2020_paper.pdf)
- 2020-CVPR-[Regularizing Class-Wise Predictions via Self-Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yun_Regularizing_Class-Wise_Predictions_via_Self-Knowledge_Distillation_CVPR_2020_paper.pdf)
- 2020-ICML-[Feature-map-level Online Adversarial Knowledge Distillation](https://arxiv.org/abs/2002.01775)
- 2020-NIPS-[Self-Distillation as Instance-Specific Label Smoothing](https://papers.nips.cc/paper/2020/hash/1731592aca5fb4d789c4119c65c10b4b-Abstract.html)
- 2020-NIPS-[Ensemble Distillation for Robust Model Fusion in Federated Learning](https://papers.nips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html)
- 2020-NIPS-[Self-Distillation Amplifies Regularization in Hilbert Space](https://papers.nips.cc/paper/2020/hash/2288f691b58edecadcc9a8691762b4fd-Abstract.html)
- 2020-NIPS-[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://papers.nips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- 2020-NIPS-[Residual Distillation: Towards Portable Deep Neural Networks without Shortcuts](https://papers.nips.cc/paper/2020/hash/657b96f0592803e25a4f07166fff289a-Abstract.html)
- 2020-NIPS-[Kernel Based Progressive Distillation for Adder Neural Networks](https://papers.nips.cc/paper/2020/hash/912d2b1c7b2826caf99687388d2e8f7c-Abstract.html)
- 2020-NIPS-[Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space](https://papers.nips.cc/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html)
- 2020-NIPS-[Task-Oriented Feature Distillation](https://papers.nips.cc/paper/2020/hash/a96b65a721e561e1e3de768ac819ffbb-Abstract.html)
- 2020-NIPS-[Comprehensive Attention Self-Distillation for Weakly-Supervised Object Detection](https://papers.nips.cc/paper/2020/hash/c3535febaff29fcb7c0d20cbe94391c7-Abstract.html)
- 2020-NIPS-[Distributed Distillation for On-Device Learning](https://papers.nips.cc/paper/2020/hash/fef6f971605336724b5e6c0c12dc2534-Abstract.html)
- 2020-NIPS-[Knowledge Distillation in Wide Neural Networks: Risk Bound, Data Efficiency and Imperfect Teacher](https://papers.nips.cc/paper/2020/hash/ef0d3930a7b6c95bd2b32ed45989c61f-Abstract.html)
- 2020.12-[Knowledge Distillation Thrives on Data Augmentation](https://arxiv.org/abs/2012.02909)
- 2020.12-[Multi-head Knowledge Distillation for Model Compression](https://arxiv.org/abs/2012.02911)

**2021**
- 2021-AAAI-[Cross-Layer Distillation with Semantic Calibration](https://arxiv.org/abs/2012.03236) [[Code](https://github.com/DefangChen/SemCKD)]
- 2021-ICLR-[Distilling Knowledge from Reader to Retriever for Question Answering](https://openreview.net/forum?id=NTEz-6wysdb)
- 2021-ICLR-[Improve Object Detection with Feature-based Knowledge Distillation: Towards Accurate and Efficient Detectors](https://openreview.net/forum?id=uKhGRvM8QNH)
- 2021-ICLR-[Knowledge distillation via softmax regression representation learning](https://openreview.net/forum?id=ZzwDy_wiWv) [[Code](https://github.com/jingyang2017/KD_SRRL)]
- 2021-ICLR-[Knowledge Distillation as Semiparametric Inference](https://openreview.net/forum?id=m4UCf24r0Y)
- 2021-ICLR-[Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study](https://openreview.net/forum?id=PObuuGVrGaZ)
- 2021-ICLR-[Rethinking Soft Labels for Knowledge Distillation: A Bias–Variance Tradeoff Perspective](https://openreview.net/forum?id=gIHd-5X324)
- 2021-CVPR-[Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation](https://arxiv.org/abs/2103.08273) [[PyTorch Code](https://github.com/MingiJi/FRSKD)]
- 2021-CVPR-[Complementary Relation Contrastive Distillation](https://arxiv.org/abs/2103.16367)
- 2021-CVPR-[Distilling Knowledge via Knowledge Review](https://arxiv.org/abs/2104.09044) [[Code](https://github.com/dvlab-research/ReviewKD)]
- 2021-ICML-[KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation]()
- 2021-ICML-[A statistical perspective on distillation]()
- 2021-ICML-[Training data-efficient image transformers & distillation through attention]()
- 2021-ICML-[Zero-Shot Knowledge Distillation from a Decision-Based Black-Box Model]()
- 2021-ICML-[Data-Free Knowledge Distillation for Heterogeneous Federated Learning]()
- 2021-ICML-[Simultaneous Similarity-based Self-Distillation for Deep Metric Learning]()
- 2021-NIPS-[Slow Learning and Fast Inference: Efficient Graph Similarity Computation via Knowledge Distillation](https://papers.nips.cc/paper/2021/file/75fc093c0ee742f6dddaa13fff98f104-Paper.pdf) [[Code](https://github.com/canqin001/Efficient_Graph_Similarity_Computation)]

#### 2022
- 2022-ECCV-[R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis](https://arxiv.org/abs/2203.17261) [Code](https://github.com/snap-research/R2L)
- 2022-NIPS-[An Analytical Theory of Curriculum Learning in Teacher-Student Networks](https://openreview.net/forum?id=4d_tnQ_agHI)


## Papers [AutoML (NAS etc.)]
- 2016.11-[Neural architecture search with reinforcement learning](https://arxiv.org/abs/1611.01578)
- 2019-CVPR-[Searching for A Robust Neural Architecture in Four GPU Hours](https://github.com/D-X-Y/GDAS/blob/master/data/GDAS.pdf) [[Code](https://github.com/D-X-Y/GDAS)]
- 2019-CVPR-[FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443)
- 2019-CVPR-[RENAS: Reinforced Evolutionary Neural Architecture Search](https://arxiv.org/abs/1808.00193)
- 2019-NIPS-[Meta Architecture Search](https://papers.nips.cc/paper/9301-meta-architecture-search)
- 2019-NIPS-[SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers](https://papers.nips.cc/paper/8743-sparse-sparse-architecture-search-for-cnns-on-resource-constrained-microcontrollers)
- 2020-NIPS-[Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation](https://papers.nips.cc/paper/2020/hash/62d75fb2e3075506e8837d8f55021ab1-Abstract.html)
- 2020-NIPS-[Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search](https://papers.nips.cc/paper/2020/hash/d072677d210ac4c03ba046120f0802ec-Abstract.html)
- 2020-NIPS-[Theory-Inspired Path-Regularized Differential Network Architecture Search](https://papers.nips.cc/paper/2020/hash/5e1b18c4c6a6d31695acbae3fd70ecc6-Abstract.html)
- 2020-NIPS-[ISTA-NAS: Efficient and Consistent Neural Architecture Search by Sparse Coding](https://papers.nips.cc/paper/2020/hash/76cf99d3614e23eabab16fb27e944bf9-Abstract.html)
- 2020-NIPS-[Semi-Supervised Neural Architecture Search](https://papers.nips.cc/paper/2020/hash/77305c2f862ad1d353f55bf38e5a5183-Abstract.html)
- 2020-NIPS-[Bridging the Gap between Sample-based and One-shot Neural Architecture Search with BONAS](https://papers.nips.cc/paper/2020/hash/13d4635deccc230c944e4ff6e03404b5-Abstract.html)
- 2020-NIPS-[Does Unsupervised Architecture Representation Learning Help Neural Architecture Search?](https://papers.nips.cc/paper/2020/hash/937936029af671cf479fa893db91cbdd-Abstract.html)
- 2020-NIPS-[Differentiable Neural Architecture Search in Equivalent Space with Exploration Enhancement](https://papers.nips.cc/paper/2020/hash/9a96a2c73c0d477ff2a6da3bf538f4f4-Abstract.html)
- 2020-NIPS-[CLEARER: Multi-Scale Neural Architecture Search for Image Restoration](https://papers.nips.cc/paper/2020/hash/c6e81542b125c36346d9167691b8bd09-Abstract.html)
- 2020-NIPS-[A Study on Encodings for Neural Architecture Search](https://papers.nips.cc/paper/2020/hash/ea4eb49329550caaa1d2044105223721-Abstract.html)
- 2020-NIPS-[Auto-Panoptic: Cooperative Multi-Component Architecture Search for Panoptic Segmentation](https://papers.nips.cc/paper/2020/hash/ec1f764517b7ffb52057af6df18142b7-Abstract.html)
- 2020-NIPS-[Hierarchical Neural Architecture Search for Deep Stereo Matching](https://papers.nips.cc/paper/2020/hash/fc146be0b230d7e0a92e66a6114b840d-Abstract.html)

*

## Papers [Interpretability]
- 2010-JMLR-[How to explain individual classification decisions](http://www.jmlr.org/papers/v11/baehrens10a.html)
- 2015-PLOS ONE-[On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](http://heatmapping.org/)
- 2015-CVPR-[Learning to generate chairs with convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)
- 2015-CVPR-[Understanding deep image representations by inverting them](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.html) [2016 IJCV version: [Visualizing deep convolutional neural networks using natural pre-images](https://link.springer.com/content/pdf/10.1007%2Fs11263-016-0911-8.pdf)]
- 2016-CVPR-[Inverting Visual Representations with Convolutional Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Dosovitskiy_Inverting_Visual_Representations_CVPR_2016_paper.html)
- 2016-KDD-["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- 2016-ICMLw-[The Mythos of Model Interpretability](https://arxiv.org/abs/1606.03490)
- 2017-NIPSw-[The (Un)reliability of saliency methods](https://arxiv.org/abs/1711.00867)
- 2017-DSP-[Methods for interpreting and understanding deep neural networks](https://arxiv.org/abs/1706.07979)
- 2018-ICML-[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors](https://arxiv.org/abs/1711.11279)
- 2018-CVPR-[Deep Image Prior](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html) [[Code](https://dmitryulyanov.github.io/deep_image_prior)]
- 2018-NIPSs-[Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)
- 2018-NIPSs-[Human-in-the-Loop Interpretability Prior](https://arxiv.org/abs/1805.11571)
- 2018-NIPS-[To Trust Or Not To Trust A Classifier](https://arxiv.org/abs/1805.11783) [[Code](https://github.com/google/TrustScore)]
- 2019-AISTATS-[Interpreting Black Box Predictions using Fisher Kernels](https://arxiv.org/abs/1810.10118)
- 2019.05-[Luck Matters: Understanding Training Dynamics of Deep ReLU Networks](https://arxiv.org/pdf/1905.13405.pdf)
- 2019.05-[Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)
- 2019.06-[The Generalization-Stability Tradeoff in Neural Network Pruning](https://arxiv.org/abs/1906.03728)
- 2019.06-[One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/abs/1906.02773)
- 2019-Book-[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/index.html)


## Workshops
- [2017-ICML Tutorial](http://people.csail.mit.edu/beenkim/icml_tutorial.html): interpretable machine learning
- [2018-ICML Workshop](https://openreview.net/group?id=ICML.cc/2018/ECA): Efficient Credit Assignment in Deep Learning and Reinforcement Learning
- CDNNRIA Workshop (Compact Deep Neural Network Representation with Industrial Applications): [1st-2018-NIPSw](https://openreview.net/group?id=NIPS.cc/2018/Workshop/CDNNRIA), [2nd-2019-ICMLw](https://sites.google.com/view/icml2019-on-device-compact-dnn)
- LLD Workshop (Learning with Limited Data): [1st-2017-NIPSw](https://lld-workshop.github.io/2017/), [2nd-2019-ICLRw](https://lld-workshop.github.io/)
- WHI (Worshop on Human Interpretability in Machine Learning): [1st-2016-ICMLw](https://sites.google.com/site/2016whi/), [2nd-2017-ICMLw](https://sites.google.com/view/whi2017/home), [3rd-2018-ICMLw](https://sites.google.com/view/whi2018)
- [NIPS-18 Workshop on Systems for ML and Open Source Software](http://learningsys.org/nips18/schedule.html)
- MLPCD Workshop (Machine Learning on the Phone and other Consumer Devices): [2nd-2018-NIPSw](https://sites.google.com/view/nips-2018-on-device-ml/home)
- [Workshop on Bayesian Deep Learning](http://bayesiandeeplearning.org/)
- [2020 CVPR Workshop on NAS](https://sites.google.com/view/cvpr20-nas/program)

## Books & Courses
- [TinyML and Efficient Deep Learning](https://efficientml.ai/) @MIT by Prof. Song Han

## Lightweight DNN Engines/APIs
- [NNPACK](https://github.com/Maratyszcza/NNPACK)
- DMLC: [Tensor Virtual Machine (TVM): Open Deep Learning Compiler Stack](https://github.com/dmlc/tvm)
- Tencent: [NCNN](https://github.com/Tencent/ncnn)
- Xiaomi: [MACE](https://github.com/XiaoMi/mace), [Mobile AI Benchmark](https://github.com/XiaoMi/mobile-ai-bench)
- Alibaba: [MNN](https://github.com/alibaba/MNN) [blog (in Chinese)](https://yq.aliyun.com/articles/707014?spm=a2c4e.11153940.0.0.696d586bavHos1)
- Baidu: [Paddle-Slim](https://github.com/PaddlePaddle/models/tree/v1.4/PaddleSlim), [Paddle-Mobile](https://github.com/PaddlePaddle/paddle-mobile), [Anakin](https://github.com/PaddlePaddle/Anakin)
- Microsoft: [ELL](https://microsoft.github.io/ELL/), AutoML tool [NNI](https://github.com/microsoft/nni)
- Facebook: [Caffe2/PyTorch](https://caffe2.ai/)
- Apple: [CoreML](https://developer.apple.com/documentation/coreml) (iOS 11+)
- Google: [ML-Kit](https://developers.google.cn/ml-kit/), [NNAPI](https://developer.android.google.cn/ndk/guides/neuralnetworks/index.html) (Android 8.1+), [TF-Lite](https://tensorflow.google.cn/lite)
- Qualcomm: [Snapdragon Neural Processing Engine (SNPE)](https://developer.qualcomm.com/software/adreno-gpu-sdk/gpu), [Adreno GPU SDK](https://developer.qualcomm.com/software/adreno-gpu-sdk/gpu)
- Huawei: [HiAI](https://developer.huawei.com/consumer/cn/hiai)
- ARM: [Tengine](https://github.com/OAID/Tengine/)
- Related: [DAWNBench: An End-to-End Deep Learning Benchmark and Competition](https://dawn.cs.stanford.edu//benchmark/index.html)

## Related Repos and Websites
- [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS)
- [Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
- [Awesome-Knowledge-Distillation](https://github.com/FLHonker/Awesome-Knowledge-Distillation)
- [MS AI-System open course](https://github.com/microsoft/AI-System/tree/main/Lectures)
- [caffe-int8-convert-tools](https://github.com/BUG1989/caffe-int8-convert-tools)
- [Neural-Networks-on-Silicon](https://github.com/fengbintu/Neural-Networks-on-Silicon)
- [Embedded-Neural-Network](https://github.com/ZhishengWang/Embedded-Neural-Network)
- [model_compression](https://github.com/j-marple-dev/model_compression)
- [model-compression](https://github.com/666DZY666/model-compression) (in Chinese)
- [Efficient-Segmentation-Networks](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)
- [AutoML NAS Literature](https://www.automl.org/automl/literature-on-neural-architecture-search/)
- [Papers with code](https://paperswithcode.com/task/network-pruning)
- [ImageNet Benckmark](https://paperswithcode.com/sota/image-classification-on-imagenet)
- [Self-supervised ImageNet Benckmark](https://paperswithcode.com/sota/self-supervised-image-classification-on)
- [NVIDIA Blog with Sparsity Tag](https://developer.nvidia.com/blog/tag/sparsity/) 
