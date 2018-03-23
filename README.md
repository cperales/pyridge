# PyRidge

[![Build Status](https://travis-ci.org/cperales/PyRidge.svg?branch=master)](https://travis-ci.org/cperales/PyRidge) [![Coverage Status](https://coveralls.io/repos/github/cperales/PyRidge/badge.svg?branch=master)](https://coveralls.io/github/cperales/PyRidge?branch=master)

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

## How to install it within a virtual environment

It is recommended to install the framework in a virtual environment

```bash
virtualenv -p python3 env
```

In order to activate the virtual environment

```bash
source env/bin/activate
```

To deactivate, just write ```deactivate```. Then, it is necessary
to install the requirements

```bash
pip install -r requirements.txt
```

To use it in any folder, you should install it as a dependency:

```bash
pip install -e .
```

or 

```bash
python setup.py install
```

## Algorithms

Right now, there are the following algorithms programmed:

* Kernel Ridge (KRidge) [1]
* Neural Ridge (NRidge) [2]
* AdaBoost Neural Ridge (AdaBoostNRidge) [3]

## Code documentation

Documentation is published [online](https://cperales.github.io/PyRidge/). It can also be compiled locally, just having
`sphinx` installed. In the main directory, run the following code:

```bash
sphinx-build docs/source docs/
```

## Data

Some data can be downloaded [here!](https://drive.google.com/file/d/1O67sgZzRtWtVUHa3qaklTsZnvEWF10Iv/view?usp=sharing).
In order to run the tests, `data` folder should be in main directory.


## An example

You can run a test for every algorithm, just simply

```bash
python test/
```

Also, there is an individual test for each algorithm,
and a test with JSON implementation

```bash
python test/test_json.py
```

By default, logging level is set to `DEBUG`.

## How to use a classifier manually

It is also useful to know how to use a classifier. In this framework, a classifier is an object with different methods,
that allows you to train from data, predict a label for test data, save the classifier...

Training a classifier, for example, a Kernel Ridge classifier:

```python
from pyridge.algorithm import KRidge
from pyridge.utils.preprocess import prepare_data

# Data
folder = 'data/newthyroid'
train_dataset = 'train_newthyroid.0'
train_data, train_target = prepare_data(folder=folder,
                                          dataset=train_dataset)
# Classifier                                        
clf = KRidge()
clf.fit(train_data=train_data, train_target=train_target)
``` 

Once trained, using the classifier to predict a label for test data is as easy as:

```python
test_dataset = 'test_newthyroid.0'
# In case in test data there are less target labels than in train data 
n_targ = train_target.shape[1]

test_data, test_target = prepare_data(folder=folder,
                                      dataset=test_dataset,
                                      n_targ=n_targ)
predicted_labels = clf.predict(test_data=test_data)
```


## Future work

- [ ] Functions to create classifiers from data, save them in as a file,
 and use them to predict, giving an understandable output.
- [ ] More examples with real predictions, not testing.
- [ ] Combine PyRidge with [pylm](http://pylm.readthedocs.io/en/latest/),
a high level queue manager in Python with communication patterns such as client-server-workers 


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
