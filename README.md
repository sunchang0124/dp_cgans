# 👯 DP-CGANS (Differentially Private - Conditional Generative Adversarial NetworkS)

[![PyPi Shield](https://img.shields.io/pypi/v/dp-cgans)](https://pypi.org/project/dp-cgans/) [![Py versions](https://img.shields.io/pypi/pyversions/dp-cgans)](https://pypi.org/project/dp-cgans/) [![Test package](https://github.com/sunchang0124/dp_cgans/actions/workflows/test.yml/badge.svg)](https://github.com/sunchang0124/dp_cgans/actions/workflows/test.yml) [![Publish package](https://github.com/sunchang0124/dp_cgans/actions/workflows/publish.yml/badge.svg)](https://github.com/sunchang0124/dp_cgans/actions/workflows/publish.yml)



<!-- [![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha) -->
<!-- [![PyPi Shield](https://img.shields.io/badge/pypi-v0.0.2-blue)](https://pypi.org/project/dp-cgans/) -->
<!-- [![Tests](https://github.com/sdv-dev/SDV/workflows/Run%20Tests/badge.svg)](https://github.com/sdv-dev/SDV/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster) -->

**Abstract**: This repository presents a Conditional Generative Adversary Networks (GANs) on tabular data (and RDF data) combining with Differential Privacy techniques. Our pre-print publication: [Generating synthetic personal health data using conditional generative adversarial networks combining with differential privacy](https://doi.org/10.1016/j.jbi.2023.104404).

**Author**: Chang Sun, Institute of Data Science, Maastricht University
**Start date**: Nov-2021
**Status**: Under development

**Note**: "Standing on the shoulders of giants". This repository is inspired by the excellent work of [CTGAN](https://github.com/sdv-dev/CTGAN) from [Synthetic Data Vault (SDV)](https://github.com/sdv-dev/SDV), [Tensorflow Privacy](https://github.com/tensorflow/privacy), and [RdfPdans](https://github.com/cadmiumkitty/rdfpandas). Highly appreciate they shared the ideas and implementations, made code publicly available, well-written documentation. More related work can be found in the References below.  

This package is extended from SDV (https://github.com/sdv-dev/SDV), CTGAN (https://github.com/sdv-dev/CTGAN), and Differential Privacy in GANs (https://github.com/civisanalytics/dpwgan). The author modified the conditional matrix and cost functions to emphasize on the relations between variables. The main changes are in ctgan/synthesizers/ctgan.py ../data_sampler.py ../data_transformer.py


## 📥️ Installation

You will need Python >=3.8+ and <3.10

```shell
pip install dp-cgans
```

## 🪄 Usage

### ⌨️ Use as a command-line interface

You can easily generate synthetic data for a file using your terminal after installing `dp-cgans` with pip.

To quickly run our example, you can download the [example data](https://raw.githubusercontent.com/sunchang0124/dp_cgans/main/resources/example_tabular_data_UCIAdult.csv):

```bash
wget https://raw.githubusercontent.com/sunchang0124/dp_cgans/main/resources/example_tabular_data_UCIAdult.csv
```

Then run `dp-cgans`:

```bash
dp-cgans gen example_tabular_data_UCIAdult.csv --epochs 2 --output out.csv --gen-size 100
```

Get a full rundown of the available options for generating synthetic data with:

```bash
dp-cgans gen --help
```

### 🐍 Use with python 

This library can also be used directly in python scripts

If your input is tabular data (e.g., csv):

 ```python
from dp_cgans import DP_CGAN
import pandas as pd

tabular_data=pd.read_csv("../resources/example_tabular_data_UCIAdult.csv")

# We adjusted the original CTGAN model from SDV. Instead of looking at the distribution of individual variable, we extended to two variables and keep their corrll
model = DP_CGAN(
    epochs=100, # number of training epochs
    batch_size=1000, # the size of each batch
    log_frequency=True,
    verbose=True,
    generator_dim=(128, 128, 128),
    discriminator_dim=(128, 128, 128),
    generator_lr=2e-4, 
    discriminator_lr=2e-4,
    discriminator_steps=1, 
    private=False,
)

print("Start training model")
model.fit(tabular_data)
model.save("generator.pkl")

# Generate 100 synthetic rows
syn_data = model.sample(100)
syn_data.to_csv("syn_data_file.csv")
 ```

<!-- 
2. If your input data is in RDF format:

  ```python
from dp_cgans import DP_CGAN
from dp_cgans import RDF_to_Tabular

# Step 1. Load RDF to a plain table (dataframe)
plain_tabular=RDF_to_Tabular(file_path="../resources/example_rdf_data.ttl")

# Step 2. Convert plain table to a structured table 
# After step 1, RDF data will be converted a plain tabular dataset (all the nodes/entities will be presented as rows. Step 2 will structure the table by recognizing and sorting the types of the entities, replacing the URI with actual value which is attached to that URI. Users can decide how many levels they want to unfold their RDF models to tabular datasets.)
tabular_data,rel_pred_obj=plain_tabular.fit_convert(user_define_data_instance="http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C16960", 
                                                    user_define_is_a=["rdf:type{URIRef}"], 
                                                    user_define_has_value=["http://www.cancerdata.org/roo/P100042"], 
                                                    set_level="full", 
                                                    as_column='object', 
                                                    remove_columns_unique_values=True)

# Step 3. Define your GANS model
model = DP_CGAN(
    epochs=100, # number of training epochs
    batch_size=1000, # the size of each batch
    log_frequency=True,
    verbose=True,
    generator_dim=(128, 128, 128),
    discriminator_dim=(128, 128, 128),
    generator_lr=2e-4, 
    discriminator_lr=2e-4,
    discriminator_steps=1, 
    private=False,
)

print("Start training model")
model.fit(tabular_data)

# Sample the generated synthetic data
model.sample(100)
  ```
-->


## 🧑‍💻 Development setup


For development, we recommend to install and use [Hatch](https://hatch.pypa.io/latest/), as it will automatically install and sync the dependencies when running development scripts. But you can also directly create a virtual environment and install the library with `pip install -e .`

### Install

Clone the repository:

```bash
git clone https://github.com/sunchang0124/dp_cgans
cd dp_cgans
```

> When working in development the `hatch` tool will automatically install and sync the dependencies when running a script. But you can also directly 

### Run

Run the library with the CLI:

```bash
hatch -v run dp-cgans gen --help
```

You can also enter a new shell with the virtual environments automatically activated:

```bash
hatch shell
dp-cgans gen --help
```

### Tests

Run the tests locally:

```bash
hatch run pytest -s
```

### Format

Run formatting and linting (black and ruff):

```bash
hatch run fmt
```

### Reset the virtual environments

In case the virtual environments is not updating as expected you can easily reset it with:

```bash
hatch env prune
```

## 📦️ New release process

The deployment of new releases is done automatically by a GitHub Action workflow when a new release is created on GitHub. To release a new version:

1. Make sure the `PYPI_API_TOKEN` secret has been defined in the GitHub repository (in Settings > Secrets > Actions). You can get an API token from PyPI [here](https://pypi.org/manage/account/).

2. Increment the `version` number in `src/dp_cgans/__init__.py` file:

   ```bash
   hatch version fix    # Bump from 0.0.1 to 0.0.2
   hatch version minor  # Bump from 0.0.1 to 0.1.0
   hatch version 0.1.1  # Bump to the specified version
   ```

3. Create a new release on GitHub, which will automatically trigger the publish workflow, and publish the new release to PyPI.

You can also manually build and publish from you laptop:

```bash
hatch build
hatch publish
```

## 📚️ References / Further reading 

There are many excellent work on generating synthetic data using GANS and other methods. We list the studies that made great conbributions for the field and inspiring for our work.

##### GANS

   1. Synthetic Data Vault (SDV) [[Paper](https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf)] [[Github](https://github.com/sdv-dev/SDV)]
   2. Modeling Tabular Data using Conditional GAN (a part of SDV) [[Paper](https://arxiv.org/abs/1907.00503)] [[Github](https://github.com/sdv-dev/CTGAN)]
   3. Wasserstein GAN [[Paper](https://arxiv.org/pdf/1701.07875.pdf)]
   4. Improved Training of Wasserstein GANs [[Paper](https://papers.nips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)]
   5. Synthesising Tabular Data using Wasserstein Conditional GANs with Gradient Penalty (WCGAN-GP) [[Paper](http://ceur-ws.org/Vol-2771/AICS2020_paper_57.pdf)]
   6. PacGAN: The power of two samples in generative adversarial networks [[Paper](https://proceedings.neurips.cc/paper/2018/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf)]
   7. CTAB-GAN: Effective Table Data Synthesizing [[Paper](https://arxiv.org/pdf/2102.08369.pdf)]
   8. Conditional Tabular GAN-Based Two-Stage Data Generation Scheme for Short-Term Load Forecasting [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9253644)]
   9. TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks [[Paper](https://arxiv.org/pdf/2109.00666.pdf)]
   10. Conditional Wasserstein GAN-based Oversampling of Tabular Data for Imbalanced Learning [[Paper](https://arxiv.org/pdf/2008.09202.pdf)]

   ##### Differential Privacy

   1. Tensorflow Privacy [[Github](https://github.com/tensorflow/privacy)]
   2. Renyi Differential Privacy [[Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46029.pdf)]
   3. DP-CGAN : Differentially Private Synthetic Data and Label Generation [[Paper](https://arxiv.org/pdf/2001.09700.pdf)]
   4. Differentially Private Generative Adversarial Network [[Paper](https://arxiv.org/pdf/1802.06739.pdf)] [[Github](https://github.com/illidanlab/dpgan)] Another implementation [[Github](https://github.com/civisanalytics/dpwgan)]
   5. Private Data Generation Toolbox [[Github](https://github.com/BorealisAI/private-data-generation)]
   6. autodp: Automating differential privacy computation [[Github](https://github.com/yuxiangw/autodp)]
   7. Differentially Private Synthetic Medical Data Generation using Convolutional GANs [[Paper](https://arxiv.org/pdf/2012.11774.pdf)]
   8. DTGAN: Differential Private Training for Tabular GANs [[Paper](https://arxiv.org/pdf/2107.02521.pdf)]
   9. DIFFERENTIALLY PRIVATE SYNTHETIC DATA: APPLIED EVALUATIONS AND ENHANCEMENTS [[Paper](https://arxiv.org/pdf/2011.05537.pdf)]
   10. FFPDG: FAST, FAIR AND PRIVATE DATA GENERATION [[Paper](https://sdg-quality-privacy-bias.github.io/papers/SDG_paper_19.pdf)]

##### Others

   1. EvoGen: a Generator for Synthetic Versioned RDF [[Paper](http://ceur-ws.org/Vol-1558/paper9.pdf)]
   2. Generation and evaluation of synthetic patient data [[Paper](https://bmcmedresmethodol.biomedcentral.com/track/pdf/10.1186/s12874-020-00977-1.pdf)]
   3. Fake It Till You Make It: Guidelines for Effective Synthetic Data Generation [[Paper](https://www.mdpi.com/2076-3417/11/5/2158)]
   4. Generating and evaluating cross-sectional synthetic electronic healthcare data: Preserving data utility and patient privacy [[Paper](https://onlinelibrary.wiley.com/doi/epdf/10.1111/coin.12427)]
   5. Synthetic data for open and reproducible methodological research in social sciences and official statistics [[Paper](https://link.springer.com/article/10.1007/s11943-017-0214-8#Sec2)]
   6. A Study of the Impact of Synthetic Data Generation Techniques on Data Utility using the 1991 UK Samples of Anonymised Records [[Paper](https://unece.org/fileadmin/DAM/stats/documents/ece/ces/ge.46/2017/4_utility_paper.pdf)]
