from setuptools import setup

setup(
    name="pyridge",
    description="Ridge Classification in Python",
    version="1.0",
    author="Carlos Perales",
    author_email="cperales@uloyola.es",
    packages=['pyridge',
              'pyridge.algorithm',
              'pyridge.utils',
              'pyridge.generic'
    ],
    zip_safe=False,
    install_requires=[],
    include_package_data=True,
    setup_requires=[],
    tests_require=[]
)

