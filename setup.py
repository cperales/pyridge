from setuptools import setup

setup(
    name="pyelm",
    description="Extreme Learning Machine in Python",
    version="0.6",
    author="Carlos Perales",
    author_email="cperales@uloyola.es",
    packages=['pyelm',
              'pyelm.algorithm',
              'pyelm.utils',
              'pyelm.generic'
    ],
    zip_safe=False,
    install_requires=[],
    include_package_data=True,
    setup_requires=[],
    tests_require=[]
)

