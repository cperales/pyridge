# PyELM

This repository contains some supervised machine learning algorithms from the family of
[Extreme Learning Machine](https://en.wikipedia.org/wiki/Extreme_learning_machine) learners,
which are a special type of feedforward neural network.

Main motivation of this repository is translating from MATLAB to Python 3 what
[I am](https://www.linkedin.com/in/carlos-perales-cperales/) doing in my PhD in Data Science
in [Universidad Loyola Andalucía](https://www.uloyola.es/investigacion/departamentos/metodos-cuantitativos).

Documentation is in progress, but some part is available [here](https://cperales.github.io/PyELM/).

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


## An example

Just run

```bash
python entry_point_ELM.py
```

## How to use a classifier manually

It is also useful to know how to use a classifier. In this framework, a classifier is an object with different methods, that allows you to train from data, predict a label for test data, save the classifier...

Training a classifier:

```python
from algorithm import *
clf = algorithm_dict[config_options['Algorithm']['name']]()
clf.set_range_param(hyperparameters)
train_dict = {'data': training_data, 'target': training_target}
cross_validation(clf, train_dict)
``` 

Once trained, using the classifier to predict a label for testing data is as easy as:

```python
predicted_labels = clf.predict(test_data=testing_data)
```

## Algorithms

Right now, there are the following algorithms programmed:

* Neural Extreme Learning Machine (NELM)
* AdaBoost Neural Extreme Learning Machine (AdaBoostNELM)

## Code documentation

Documentation can be compiled locally in linux. In the main directory, run the following code:

```bash
sphinx-build docs/source docs/
```
