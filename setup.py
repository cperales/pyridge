from setuptools import setup

setup(
    name="pyridge",
    description="Python Ridge-ELM Classifier",
    version="2.0",
    author="Carlos Perales",
    author_email="sir.perales@gmail.com",
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
                      'scipy'
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
