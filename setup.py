"""
Setup file for py-tidymodels package
"""

from setuptools import setup, find_packages

setup(
    name="py-tidymodels",
    version="1.0.0",
    description="Python port of R's tidymodels ecosystem for time series forecasting and machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Matthew Deane",
    author_email="matthew.deane@example.com",
    url="https://github.com/m-deane/py-tidymodels2",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "reference", "_md", "_guides", "docs", "my-docs"]),
    package_data={
        "py_agent": ["knowledge/*.json"],
    },
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.3.0",
        "numpy>=2.2.0",
        "patsy>=1.0.0",
        "scikit-learn>=1.7.0",
        "prophet>=1.2.0",
        "statsmodels>=0.14.0",
        "scipy>=1.14.0",
        "matplotlib>=3.10.0",
        "plotly>=6.3.0",
        "skforecast>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.0",
            "pytest-cov>=7.0.0",
            "jupyter>=1.1.0",
            "notebook>=7.3.0",
            "ipykernel>=7.0.0",
            "black>=25.0.0",
            "flake8>=7.3.0",
            "mypy>=1.18.0",
        ],
        "boosting": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
        ],
        "advanced": [
            "pygam>=0.9.0",
            "pmdarima>=2.0.0",
        ],
        "agent": [
            "anthropic>=0.40.0",
            "chromadb>=0.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="tidymodels time-series forecasting machine-learning arima prophet xgboost modeling workflow",
)
