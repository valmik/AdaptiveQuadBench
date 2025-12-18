"""
Setup script for AdaptiveQuadBench package.
"""

from setuptools import setup, find_packages

setup(
    name="aqb",
    version="0.1.0",
    author="AdaptiveQuadBench Contributors",
    url="https://github.com/Dz298/AdaptiveQuadBench",
    packages=find_packages("."),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "seaborn",
        "pyquaternion",
        "gymnasium",
        "tqdm",
        "coloredlogs",
        "filterpy",
    ],
)

