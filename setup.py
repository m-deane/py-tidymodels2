"""
Setup file for py-tidymodels package
"""

from setuptools import setup, find_packages

setup(
    name="py-tidymodels",
    version="0.1.0",
    description="Python port of R's tidymodels ecosystem for time series regression and forecasting",
    author="Matthew Deane",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "reference"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.3.3",
        "numpy>=2.2.6",
        "patsy>=1.0.2",
        "scikit-learn>=1.7.2",
        "prophet>=1.2.1",
        "statsmodels>=0.14.5",
        "scipy>=1.14.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.2",
            "pytest-cov>=7.0.0",
            "jupyter>=1.1.1",
            "notebook>=7.3.2",
        ],
    },
)
