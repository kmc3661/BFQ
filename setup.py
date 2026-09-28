#!/usr/bin/env python
from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bfq",
    version="0.1.0",
    description="BFQ: Balanced Fitting Quantization for Large Vision-Language Models",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
