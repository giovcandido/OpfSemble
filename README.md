# OPFsemble

Ensemble pruning involves employing different strategies to select the most relevant classifiers provided within a stacking generalization model. Among the various strategies aimed to optimize ensemble models in classification tasks, clustering approaches emerge to provide the grouping of similar classifiers based on prediction in the validation set. Further, the inference stage involves selecting a subset of classifiers from different clusters. By doing so, we ensure diversity while maintaining accuracy in the test data.

This code aims to provide several approaches for addressing the pruning of ensemble models based on the unsupervised version of the Optimum-Path Forest (OPF) algorithm. The unsupervised OPF is a graph-based clustering strategy designed to produce groups of similar data. The rationale behind this proposal is to take the classifiers as nodes in a graph and cluster them based on their similar predictions modeled as a feature matrix represented by the hits and misses of each classifier in a validation set. Each cluster is centered by a graph node named 'prototype'. After clustering the classifiers, we can take the ones representing the prototypes to perform the inference and take the majority vote as the final classification. In addition, we provide various additional strategies to perform the pruning from the produced clusters.

# Source-code structure

The code is simple and structured as follows:

```
-opfsemble
  |_ensemble.py
  |_ensemble_item.py
  |_opf_ensemble.py
  |_test.py
```

- **ensemble.py**: implements the class that encompasses the list of classifiers.
- **ensemble_item.py**: class representing an item in **ensemble.py**
- **opf_ensemble.py**: the class that implements the clustering and pruning of the classifiers within the ensemble model.
- **test.py**: a simple example including the instructions to use the provided code.

# Pre-requisites

To ensure the code performs properly, you need to install the following packages:
- opfython
- python >= 3.6
- scikit-learn>=0.24.1
- pandas >= 1.0.5
- numpy >= 1.18.5

*Obs: OPFython is a Python-inspired implementation of the OPF framework. It provides the models for supervised, semi-supervised, and unsupervised classification. You must install it to load the unsupervised OPF included in the OPFsemble implementation. Check the following link for further information:* <br>
[https://github.com/gugarosa/opfython](https://github.com/gugarosa/opfython)

# Citation

The provided implementation is based on the following article:

**Jodas, D. S., Passos, L. A., Rodrigues, D., Lucas, T. J., Da Costa, K. A. P., & Papa, J. P. (2023, June). OPFsemble: An Ensemble Pruning Approach via Optimum-Path Forest. In 2023 *30th International Conference on Systems, Signals and Image Processing (IWSSIP)* (pp. 1-5). IEEE.**

If you wish to include the OPFsemble as a baseline for comparison purposes in your research, you agree to cite the abovementioned article in all publications using this repository.

# Additional info

You are invited to check the website of the Recogna Laboratory so you can stay informed on the latest works in several domains of machine learning research:

Recogna Laboratory: [https://www.recogna.tech](https://www.recogna.tech) <br>

# Contact

If you have any further questions, please do not hesitate to contact us.

| Author                    | E-mail                        |
| ----------------------    | ----------------------        |
| Danilo Samuel Jodas       | danilojodas@gmail.com         | <br>
| Leandro Aparecido Passos  | leandro.passos@unesp.br      | <br>
| João Paulo Papa           | joao.papa@unesp.br            | <br>