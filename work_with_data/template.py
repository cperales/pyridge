# KELM
kernel_json_template = {
    "Data": {
        "dataset": ".",
        "folder": ".",
        "standarized": False
    },
    "Algorithm": {
        "name": "KELM",  # KSVM, NELM, NELM
        "hyperparameters": {
            "type": "CCA",
            "C": [0.01, 0.1, 1, 10, 100],
            "k": [0.01, 0.1, 1, 10, 100],
            "kernelFun": "rbf",
            "type": "kernel"
        }
    },
    "Report": {
        "folder": "experiments/",
        "report_name": "experiment",
        "metrics": [
            "accuracy"
        ]
    }
}

kernel_config_name = 'KELM_multitest.json'
kernel_tuple = kernel_config_name, kernel_json_template

# NELM
neural_json_json_template = {
    "Data": {
        "dataset": ".",
        "folder": ".",
        "standarized": False
    },
    "Algorithm": {
        "name": "NELM",  # KSVM, NELM, NELM
        "hyperparameters": {
            "C": [0.01, 0.1, 1, 10, 100],
            "hiddenNeurons": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                200
            ],
            "neuronFun": "sigmoid",
            "type": "neuralNetwork"
        }
    },
    "Report": {
        "folder": "experiments/",
        "report_name": "experiment",
        "metrics": [
            "accuracy"
        ]
    }
}

neural_config_name = 'NELM_multitest.json'
neural_tuple = neural_config_name, neural_json_json_template

# AdaBoost
adaboost_json_template = {
    "Data": {
        "dataset": ".",
        "folder": ".",
        "standarized": False
    },
    "Algorithm": {
        "name": "AdaBoostNELM",
        "hyperparameters": {
            "C": [0.01, 0.1, 1, 10, 100],
            "hiddenNeurons": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                200
            ],
            "ensembleSize": 5,
            "neuronFun": "sigmoid",
            "type": "neuralNetwork"
        }
    },
    "Report": {
        "folder": "experiments/",
        "report_name": "experiment",
        "metrics": [
            "accuracy",
            "diversity"
        ]
    }
}

adaboost_config_name = 'AdaBoostNELM_multitest.json'

adaboost_tuple = adaboost_config_name, adaboost_json_template

json_templates = {'AdaBoostNELM': adaboost_tuple,
                  'NELM': neural_tuple,
                  'KELM': kernel_tuple}
