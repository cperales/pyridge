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
