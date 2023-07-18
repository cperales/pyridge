# PyRidge

[![Build Status](https://app.travis-ci.com/cperales/pyridge.svg?branch=master)](https://app.travis-ci.com/github/cperales/pyridge)
[![Coverage Status](https://coveralls.io/repos/github/cperales/pyridge/badge.svg?branch=master)](https://coveralls.io/github/cperales/pyridge?branch=master)

This repository contains some supervised machine learning algorithms from the family of Ridge Classification,
also known as
[Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization) or 
[Extreme Learning Machine](https://en.wikipedia.org/wiki/Extreme_learning_machine).
A nice discussion about these terms can be seen in [this discussion in
StackExchange](https://stats.stackexchange.com/questions/234280/is-tikhonov-regularization-the-same-as-ridge-regression). 

Although ELM is a polemic topic,
due to the accusations of plagiarism (see more [here](https://github.com/scikit-learn/scikit-learn/pull/10602) and
[here](https://www.reddit.com/r/MachineLearning/comments/34y2nk/the_elm_scandal_a_formal_complaint_launched/)),
some actual research is done by applying ensemble techniques to Ridge Classification ([3, 4]), thus some some papers
are used for implementing algorithms.

Main motivation of this repository is collecting the ML algorithms from my research while
[I was](https://www.linkedin.com/in/carlos-perales-cperales/) doing my PhD in Data Science
in [Universidad Loyola Andalucía](https://www.uloyola.es/en/research/departments/quantitative-methods-department).

Documentation and examples are in progress, but some part is available [here](https://cperales.github.io/pyridge/).

## Install

To install the library along with the dependencies,

```bash
python setup.py install
```

## How to use a virtual environment

It is recommended to install the framework in a virtual environment

```bash
virtualenv -p python3.9 env
```

In order to activate the virtual environment

```bash
source env/bin/activate
```

To deactivate, just write ```deactivate```.


## Algorithms

List of algorithms implemented:

### Kernel algorithms (or Ridge classification)

* Kernel Ridge (KRidge) [1]

### Neural algorithms (or ELM classification)
* ELM [2]
* AdaBoost ELM (AdaBoostELM) [3]
* AdaBoost Negative Correlation (AdaBoostNCELM) [4]
* Bagging ELM (BaggingELM) [5]
* Diverse ELM (DiverseELM) [6]
* Regularized Ensemble ELM (RegularizedEnsembleELM) [7]
* PCA ELM [10]
* PCA LDA ELM [11]
* Sobol ELM [12]
* Paralell Layer ELM [13]
* Boosting Ridge ELM [14]


### Negative Correlation algorithms
* Negative Correlation ELM [8]
* Negative Corelation with Neural Networks [9]


## Code documentation

Documentation is published [online](https://cperales.github.io/PyRidge/). It can also be compiled locally, just having
`sphinx` installed. In the main directory, run the following code:

```bash
sphinx-build docs/source docs/
```

## Data

Some data can be downloaded [here for classification](https://www.dropbox.com/s/c5p3fg0x8selqqv/data.zip)  and
[here for regression](https://www.dropbox.com/s/6en0h3pxv8pbaj5/data_regression.zip).
In order to run the tests, `data` and `data_regression` folders should be in main directory.

Also, repository [uci-download-process](https://github.com/cperales/uci-download-process)
could help you to download some examples from [UCI dataset](https://archive.ics.uci.edu/ml/datasets.html).

## An example

You can run a test for every algorithm, just simply

```bash
pytest
```

By default, logging level is set to `INFO`.


## Bibliography

[1]: S. An, W. Liu and S. Venkatesh, "Face Recognition Using Kernel Ridge
Regression," 2007 IEEE Conference on Computer Vision and Pattern Recognition,
Minneapolis, MN, 2007, pp. 1-7.

[2]: G.-B. Huang, H. Zhou, X. Ding, and R. Zhang, “Extreme learning machine
for regression and multiclass classification,” IEEE Trans. Syst. man, Cybern.
Part B, Cybern., vol. 42, no. 2, pp. 513–29, 2012.

[3]: A. Riccardi, F. Fernández-Navarro, S. Carloni, F. Fernandez-Navarro,
and S. Carloni, “Cost-sensitive AdaBoost algorithm for ordinal regression
based on extreme learning machine,” IEEE Trans. Cybern., vol. 44, no. 10,
pp. 1898–1909, 2014.

[4]: Wang, S., Chen, H., & Yao, X. (2010, July). Negative correlation
learning for classification ensembles. In Neural Networks (IJCNN),
The 2010 International Joint Conference on (pp. 1-8). IEEE.

[5]: Tian, H., & Meng, B. (2010, September). A new modeling method based
on bagging ELM for day-ahead electricity price prediction. In Bio-Inspired
Computing: Theories and Applications (BIC-TA), 2010 IEEE Fifth
International Conference on (pp. 1076-1079). IEEE.

[6]: Perales-González, C., Carbonero-Ruz, M., Becerra-Alonso, D., &
Fernández-Navarro, F. (2018, June). A Preliminary Study of Diversity
in Extreme Learning Machines Ensembles. In International Conference
on Hybrid Artificial Intelligence Systems (pp. 302-314). Springer, Cham.

[7]: Perales-González, C., Carbonero-Ruz, M., Becerra-Alonso, D.,
Pérez-Rodríguez, F., & Fernández-Navarro, F. 
(2019, June). Regularized Ensemble Neural Networks models in the
Extreme Learning Machine framework. In Neurocomputing (DOI: 10.1016/j.neucom.2019.06.040)

[8]: Perales-González, C., Carbonero-Ruz, M., Pérez-Rodríguez, J.,
Becerra-Alonso, D., & Fernández-Navarro, F. 
(2020, March). Negative correlation learning in the extreme learning machine framework.
In Neural Comput & Applic (DOI: 10.1007/s00521-020-04788-9)

[9]: Wang, S., Chen, H., & Yao, X. (2010, July). 
Negative correlation learning for classification ensembles. 
In The 2010 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.

[10]: Castaño, A., Fernández-Navarro, F., & Hervás-Martínez, C. (2013).
PCA-ELM: a robust and pruned extreme learning machine approach
based on principal component analysis.
Neural processing letters, 37(3), 377-392.

[11]: Castaño, A., Fernández-Navarro, F., Riccardi, A., & Hervás-Martínez, C. (2016).
Enforcement of the principal component analysis–extreme learning machine
algorithm by linear discriminant analysis.
Neural Computing and Applications, 27(6), 1749-1760.

[12]: Cervellera, C., & Macciò, D. (2015). 
Low-discrepancy points for deterministic assignment of hidden 
weights in extreme learning machines. IEEE transactions on neural networks 
and learning systems, 27(4), 891-896.

[13]: Henríquez, P. A., & Ruz, G. A. (2017). 
Extreme learning machine with a deterministic assignment of hidden 
weights in two parallel layers. Neurocomputing, 226, 109-116.

[14]: Yangjun, R., Xiaoguang, S., Huyuan, S., Lijuan, S., & Xin, W. (2012, June).
Boosting ridge extreme learning machine.
2012 IEEE Symposium on Robotics and Applications (ISRA) (pp. 881-884). IEEE.
