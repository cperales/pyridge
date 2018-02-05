# PyELM

This repository contains some supervised machine learning algorithms from the family of
[Extreme Learning Machine](https://en.wikipedia.org/wiki/Extreme_learning_machine) learners,
which are a special type of feedforward neural network.

Main motivation of this repository is translating from MATLAB to Python 3 what
[I am](https://www.linkedin.com/in/carlos-perales-cperales/) doing in my PhD in Data Science
in [Universidad Loyola Andalucía](https://www.uloyola.es/investigacion/departamentos/metodos-cuantitativos).

Documentation and examples are in progress, but some part is available [here](https://cperales.github.io/PyELM/).

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

To use it in any folder, you sould install it as a dependency:

```bash
pip install -e .
```

or 

```bash
python setup.py install
```

## Algorithms

Right now, there are the following algorithms programmed:

* Neural Extreme Learning Machine (NELM) [1]
* Kernel Extreme Learning Machine (KELM) [1]
* AdaBoost Neural Extreme Learning Machine (AdaBoostNELM) [2]

## Code documentation

Documentation is published [online](https://cperales.github.io/PyELM/). It can also be compiled locally, just having
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
python test/test_json.py
```

Also, there a test implementation, which mixes with JSON

```bash
python test/test_KELM.py
```

By default, logging level is set to `DEBUG`.

## How to use a classifier manually

It is also useful to know how to use a classifier. In this framework, a classifier is an object with different methods, that allows you to train from data, predict a label for test data, save the classifier...

Training a classifier:

```python
from algorithm import *
clf = algorithm_dict[config_options['Algorithm']['name']]()
clf.set_cv_range(hyperparameters)
cross_validation(clf, train_data=train_data, train_target=train_target)
``` 

Once trained, using the classifier to predict a label for test data is as easy as:

```python
predicted_labels = clf.predict(test_data=test_data)
```

## Bibliography

[1]: G.-B. Huang, H. Zhou, X. Ding, and R. Zhang, “Extreme learning machine
for regression and multiclass classification,” IEEE Trans. Syst. man, Cybern.
Part B, Cybern., vol. 42, no. 2, pp. 513–29, 2012.

[2] A. Riccardi, F. Fernández-Navarro, S. Carloni, F. Fernandez-Navarro,
and S. Carloni, “Cost-sensitive AdaBoost algorithm for ordinal regression
based on extreme learning machine,” IEEE Trans. Cybern., vol. 44, no. 10,
pp. 1898–1909, 2014.
