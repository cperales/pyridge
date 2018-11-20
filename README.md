# PyRidge

[![Build Status](https://travis-ci.org/cperales/pyridge.svg?branch=master)](https://travis-ci.org/cperales/pyridge)
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

Main motivation of this repository is translating from MATLAB to Python 3 what
[I am](https://www.linkedin.com/in/carlos-perales-cperales/) doing in my PhD in Data Science
in [Universidad Loyola Andalucía](https://www.uloyola.es/en/research/departments/quantitative-methods-department).

Documentation and examples are in progress, but some part is available [here](https://cperales.github.io/PyRidge/).

## Install

To install the library along with the dependencies,

```bash
python setup.py install
```

## How to use a virtual environment

It is recommended to install the framework in a virtual environment

```bash
virtualenv -p python3 env
```

In order to activate the virtual environment

```bash
source env/bin/activate
```

To deactivate, just write ```deactivate```.


## Algorithms

Right now, there are the following algorithms programmed:

### Kernel algorithms (or Ridge classification)

* Kernel Ridge (KRidge) [1]

### Neural algorithms (or ELM classification)
* ELM [2]
* AdaBoost ELM (AdaBoostELM) [3]
* AdaBoost Negative Correlation (AdaBoostNCELM) [4]
* Bagging ELM (BaggingELM) [5]
* Diverse ELM (DiverseELM) [6]

## Code documentation

Documentation is published [online](https://cperales.github.io/PyRidge/). It can also be compiled locally, just having
`sphinx` installed. In the main directory, run the following code:

```bash
sphinx-build docs/source docs/
```

## Data

Some data can be downloaded [here!](https://drive.google.com/file/d/1O67sgZzRtWtVUHa3qaklTsZnvEWF10Iv/view?usp=sharing).
In order to run the tests, `data` folder should be in main directory.

Also, repository [uci-download-process](https://github.com/cperales/uci-download-process)
could help you to download some examples from [UCI dataset](https://archive.ics.uci.edu/ml/datasets.html).

## An example

You can run a test for every algorithm, just simply

```bash
python test/test_coverage.py
```

By default, logging level is set to `DEBUG`.


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
