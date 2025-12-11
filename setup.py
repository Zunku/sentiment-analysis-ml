from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # agrega dependencias si ocupas
)

# pip install -e . Crea un link simbolico al codigo, asi se actualiza solo mientras desarrollas