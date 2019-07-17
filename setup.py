from setuptools import setup

setup(
    name="pyridge",
    description="Python Ridge-ELM Classifier",
    version="3.0",
    author="Carlos Perales",
    author_email="cperales@uloyola.esm",
    packages=['pyridge',
              'pyridge.experiment',
              'pyridge.util',
              'pyridge.generic',
              'pyridge.kernel',
              'pyridge.neural'
    ],
    zip_safe=False,
    install_requires=['numpy',
                      'pandas',
                      'sklearn',
                      'scipy',
                      'pymongo',
                      'cvxopt'
                      ],
    include_package_data=True,
    setup_requires=[],
    tests_require=[],
    extras_require={
        'docs': [
            'sphinx'
        ]
    },
)
