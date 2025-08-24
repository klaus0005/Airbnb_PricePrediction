# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="airbnb_price_prediction",
    version="0.1.0",
    description="Multimodal Airbnb price prediction pipeline (tabular, text, spatial, image)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Claudia Panainte",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "jupyterlab>=3.0",
        "ipykernel>=6.0",
        "pytest>=7.0",
        "xgboost>=1.5",
        "nltk>=3.6",
        # Add more as needed
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "pytest-cov"
        ],
        "notebook": [
            "jupyter",
            "notebook"
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
