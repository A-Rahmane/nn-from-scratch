from setuptools import setup, find_packages

setup(
    name="nn-from-scratch",
    version="0.1.0",
    description="Neural Networks implemented from scratch",
    author="MENOUER Abderrahmane",
    author_email="menouer.a.rahmane@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
