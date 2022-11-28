# Allendigger

Allendigger serves as database to provide easy access to expression data with structure annotation from [Allen Brain Atlas](https://portal.brain-map.org/) and simultaneously as data analysis tool to achieve data visualization, spatial heterogeneity characterization and cell resgistration. Allendigger will be useful if you want to explore the gene expression pattern in different brain regions and reveal the spatial location of single cells via their gene expression.

## Introduction

Spatial transcriptomics is helpful to capture tissue architecture and cell spatial organization which has facilitated further understandings of biological process including developmental biology, cancer and neuroscience. However, compared to its technical challenge and immature data analysis method, ABA provides a great source for spatial expression data of mouse brains from E11.5 to Adult developing period. While the portal developed to query its data is not very handy to biologists, Allendigger allows more friendly access to visualize the spatial expression data, deciphers spatial heterigeneity of brain, and register cells to anatomical brain regions by single cell expression data.
![figure]()

## Dependencies

The project was implemented in Python 3.9. The following packages are needed for running the models and performing the analysis:
- numpy, pandas, anndata, scanpy, scipy, scikit-learn
- torch
- matplotlib, seaborn

## Installation

download the compressed file from **Release module** and use pip install

    pip install PATH/TO/FILE/allendigger-1.0.tar.gz

## Citation
