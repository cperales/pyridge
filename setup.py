from setuptools import setup

setup(
    name="pyridge",
    description="Python Ridge-ELM Classifier",
    version="3.1",
    author="Carlos Perales",
    author_email="cperales@uloyola.es",
    packages=['pyridge',
              'pyridge.experiment',
              'pyridge.preprocess',
              'pyridge.util',
              'pyridge.generic',
              'pyridge.kernel',
              'pyridge.neural',
              'pyridge.linear',
              'pyridge.negcor',
    ],
    zip_safe=False,
    install_requires=['numpy',
                      'pandas',
                      'sklearn',
                      'scipy',
                      'pymongo',
                      'cvxopt',
                      'pyscenarios',
                      'toolz',
                      ],
    include_package_data=True,
    setup_requires=[],
    tests_require=['pytest'],
    extras_require={
        'docs': [
            'sphinx'
        ]
    },
)
