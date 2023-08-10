# qpf-bic
Quantum pre-processing filter for binary image classification

This repository contains MATLAB scripts for performing binary image classification using a quantum pre-processing filter (QPF) and neural networks.  Four datasets are used as follows:

- MNIST (http://yann.lecun.com/exdb/mnist/)
- EMNIST (https://www.westernsydney.edu.au/icns/resources/reproducible_research3/publication_support_materials2/emnist)
- CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
- GTSRB (https://benchmark.ini.rub.de/)

Two scenarios are considered.  The first scenario uses all data available.  The second scenario uses 100 samples (80 samples for training and 20 samples for testing) for each trial and 100 trials are performed.

The following MATLAB scripts are provided:

- classify_all_data_mnist_nn.m: Binary classification by NN against MNIST dataset using all data.
- classify_all_data_emnist_nn.m: Binary classification by NN against EMNIST dataset using all data.
- classify_all_data_cifar_nn.m: Binary classification by NN against CIFAR-10 dataset using all data.
- classify_all_data_gtsrb_nn.m: Binary classification by NN against GTSRB dataset using all data.
- classify_all_data_mnist_qpf.m: Binary classification by QPF-NN against MNIST dataset using all data.
- classify_all_data_emnist_qpf.m: Binary classification by QPF-NN against EMNIST dataset using all data.
- classify_all_data_cifar_qpf.m: Binary classification by QPF-NN against CIFAR-10 dataset using all data.
- classify_all_data_gtsrb_qpf.m: Binary classification by QPF-NN against GTSRB dataset using all data.
- classify_100_sample_mnist_nn.m: Binary classification by NN against MNIST dataset using 100 samples and 100 trials.
- classify_100_sample_emnist_nn.m: Binary classification by NN against EMNIST dataset using 100 samples and 100 trials.
- classify_100_sample_cifar_nn.m: Binary classification by NN against CIFAR-10 dataset using 100 samples and 100 trials.
- classify_100_sample_gtsrb_nn.m: Binary classification by NN against GTSRB dataset using 100 samples and 100 trials.
- classify_100_sample_mnist_qpf.m: Binary classification by QPF-NN against MNIST dataset using 100 samples and 100 trials.
- classify_100_sample_emnist_qpf.m: Binary classification by QPF-NN against EMNIST dataset using 100 samples and 100 trials.
- classify_100_sample_cifar_qpf.m: Binary classification by QPF-NN against CIFAR-10 dataset using 100 samples and 100 trials.
- classify_100_sample_gtsrb_qpf.m: Binary classification by QPF-NN against GTSRB dataset using 100 samples and 100 trials.

In addition, the following MATLAB scripts are provided:

- map_accuracy_all_data_mnist.nn.m: Map classification accuracy by NN against each pairs of different classes from MNIST dataset using all data.
- map_accuracy_all_data_mnist.qpf.m: Map classification accuracy by QPF-NN against each pairs of different classes from MNIST dataset using all data.
- map_accuracy_all_data_gtsrb.nn.m: Map classification accuracy by NN against each pairs of different classes from GTSRB dataset using all data.
- map_accuracy_all_data_gtsrb.qpf.m: Map classification accuracy by QPF-NN against each pairs of different classes from GTSRB dataset using all data.
- plot_accuracy_100_sample_mnist.m: Plot classification accuracy by NN and by QPF-NN averaged over all different pairs of classes from MNIST dataset using 100 samples for each trial.
- plot_accuracy_100_sample_gtsrb.m: Plot classification accuracy by NN and by QPF-NN averaged over all different pairs of classes from GTSRB dataset using 100 samples for each trial.

MATLAB scripts have been tested using MATLAB 2021b running on Windows PCs.  Deep Learning Toolbox is required.
