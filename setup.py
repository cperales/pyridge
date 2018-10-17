from setuptools import setup

setup(
    name="pyridge",
    description="Python Ridge Classifier",
    version="2.0",
    author="Carlos Perales",
    author_email="cperales@uloyola.es",
    packages=['pyridge',
              'pyridge.experiment',
              'pyridge.util',
              'pyridge.generic',
              'pyridge.kernel',
              'pyridge.neural'
    ],
    zip_safe=False,
    install_requires=['numpy',
                      'sklearn',
                      'scipy'],
    include_package_data=True,
    setup_requires=[],
    tests_require=[]
)
