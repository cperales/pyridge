# Library for testing machine learning algorithms

This repository contains some metrics and configuration in order to test, at the moment, supervised classifiers.

Main motivation of this repository is translating from MATLAB to Python 3 what [I am](https://www.linkedin.com/in/carlos-perales-cperales/) doing in my PhD in Data Science in [Universidad Loyola Andalucia](https://www.uloyola.es/investigacion/departamentos/metodos-cuantitativos).

## How to install virtualenv

It is recommended to install the framework in a virtual environment

```
virtualenv -p python3 env
```

In order to activate the virtual environment

```
source env/bin/activate
```

To deactivate, just write ```deactivate```. Then, it is necessary
to install the requirements

```
pip install -r requirements.txt
```

## An example

Just run

```
python entry_point_ELM.py
```

## How to use a classifier manually

It is also useful to know how to use a classifier. In this framework, a classifier is an object with different methods, that allows you to train from data, predict a label for test data, save the classifier...

Training a classifier:

```python
from algorithm import *
clf = algorithm_dict[config_options['Algorithm']['name']]()
clf.set_conf(hyperparameters)
train_dict = {'data': training_data, 'target': training_target}
clf.config(train=train_dict)
``` 

Once trained, using the classifier to predict a label for testing data is as easy as:

```python
predicted_labels = clf.predict(test_data=testing_data)
```
