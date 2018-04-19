from setuptools import setup

setup(
   name="mcc4mcc",
   version="0.1",
   description="Model Checker Collection for the Model Checking Contest",
   author="Alban Linard",
   author_email="alban@linard.fr",
   packages=["mcc4mcc"],
   install_requires=[
       "argparse",
       "docker",
       "frozendict",
       "numpy",
       "pandas",
       "pycodestyle",
       "pylint",
       "scikit_learn",
       "tqdm",
       "xmltodict",
   ],
)
