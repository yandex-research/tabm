# TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling<!-- omit in toc -->

> [!IMPORTANT]
> The official announcement is coming soon.

:scroll: [arXiv](https://arxiv.org/abs/2410.24210)
<!-- &nbsp; :computer: [Example](./example.ipynb) -->
&nbsp; :books: [Other tabular DL projects](https://github.com/yandex-research/rtdl)

*TL;DR: TabM is a simple and powerful tabular DL architecture that efficiently imitates an ensemble of MLPs.*

> [!TIP]
> For a quick overview of the paper, see **the abstract, Figure 1 and Page 7** in the [PDF](https://arxiv.org/pdf/2410.24210).

---

Table of contents
- [Overview](#overview)
  - [Models](#models)
  - [Hyperparameters](#hyperparameters)
  - [Metrics](#metrics)
- [Set up the environment](#set-up-the-environment)
  - [Software](#software)
  - [Data](#data)
  - [Quick test](#quick-test)
- [Running the code](#running-the-code)
  - [Code overview](#code-overview)
  - [Common guidelines](#common-guidelines)
  - [Training](#training)
  - [Hyperparameter tuning](#hyperparameter-tuning)
  - [Evaluation](#evaluation)
  - [Ensembling](#ensembling)
  - [Automating all of the above](#automating-all-of-the-above)
- [Adding new datasets](#adding-new-datasets)
- [How to cite](#how-to-cite)

---

# Overview

The repository provides:
- The official implementation of the TabM and MLP models.
- The code for training and hyperparameter tuning used in the paper.
- Hyperparameter tuning spaces, tuned hyperparameters and metrics of the models on all 40+ datasets
  used in the paper.

<!-- ## Example

The `example.ipynb` notebook provides a standalone example of training TabM. -->

## Models

The following models are available in this repository.

> [!TIP]
> Among the TabM models, the following variations are recommended as a starting point:
> - $\mathrm{TabM}$ as the basic version.
> - $\mathrm{TabM_{mini}^\dagger}$ as the advanced version.

| Name                           | Comment                                        |
| :----------------------------- | :--------------------------------------------- |
| $\mathrm{MLP}$                 | The plain multilayer perceptron (MLP)          |
| $\mathrm{MLP^\dagger}$         | MLP with the piecewise-linear embeddings       |
| $\mathrm{MLP{-}PLR}$           | MLP with the periodic embeddings               |
| $\mathrm{MLP{-}PLR_{lite}}$    | MLP with the lite periodic embeddings          |
| $\mathrm{TabM_{mini}}$         | TabM-mini                                      |
| $\mathrm{TabM}$                | TabM                                           |
| $\mathrm{TabM_{mini}^\dagger}$ | TabM-mini with the piecewise-linear embeddings |
| $\mathrm{TabM^\dagger}$        | TabM with the piecewise-linear embeddings      |

## Hyperparameters

This section covers default hyperparameters and hyperparameter tuning.

**Default hyperparameters**

While there are no "official" default hyperparameters, the available tuned hyperparameters
on 40+ dataset allow obtaining a reasonable configuration for the first run.

<details>
<summary>Show how</summary>

```python
import json
from pathlib import Path

import pandas as pd

model = 'tabm'  # Or any other model from the exp/ directory.

# Load all training runs.
df = pd.json_normalize([
    json.loads(x.read_text())
    for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
])
print(df.shape)  # (1290, 181)
df.head()

def get_dataset_name(dataset_path: str) -> str:
    """
    >>> get_dataset_name('data/california')
    'california'
    >>> get_dataset_name('data/regression-num-large-0-year')
    'year'
    """
    name = dataset_path.removeprefix('data/')
    return (
        name.split('-', 4)[-1]  # The "why" benchmark.
        if name.startswith(('classif-', 'regression-'))
        else name
    )


df['Dataset'] = df['config.data.path'].map(get_dataset_name)

# The hyperparameters.
hyperparameters = [
    'config.model.k',
    'config.model.backbone.n_blocks',
    'config.model.backbone.d_block',
    'config.model.backbone.dropout',
    'config.optimizer.lr',
    'config.optimizer.weight_decay',
]

# When summarizing hyperparameters (but not metrics),
# it is enough to keep only one seed per dataset.
dfh = df.loc[df['config.seed'] == 0, ['Dataset', *hyperparameters]]

# Add additional "hyperparameters".
dfh['has_dropout'] = (dfh['config.model.backbone.dropout'] > 0).astype(float)
dfh['has_weight_decay'] = (dfh['config.optimizer.weight_decay'] > 0).astype(float)

# Some datasets have multiple splits, so they must be aggregated first.
dfh = dfh.groupby('Dataset').mean()

# Finally, compute the statistics.
# NOTE: it is important to take all statistics into account, especially the quantiles,
# not only the mean value, because the latter is not robust to outliers.
dfh.describe()
```

**Additional notes**

First, the above approach is not expected to result in a universally powerful configuration.
Generally, the more robust to hyperparameters the model is, the higher is the chance
to compose a configuration that will be a reasonable starting point on a larger number of datasets.

Second, when computing the above statistics,
a seemingly natural idea is to use only those datasets that are more similar to the task at hand.
Also, not all used datasets are equally representative for the real world usage.
However, deciding if a given dataset is relevant for the target task is not trivial.
For example, generally, filtering datasets by size may not be a reliable criteria.

One idea is to split datasets into different groups,
compute statistics separately for each group, and thus get better intuition on hyperparameters.
Examples of groups:
- By split type: "datasets with random splits" and "datasets with domain-aware splits"
  (this separation is used in the paper).
- By GBDT/DL-friendliness: "datasets where GBDT performs well" and "datasets where DL performs well".
- etc.

</details>

Based on the above approach, the following configurations can be suggested as a starting point
for TabM with the AdamW optimizer:

> [!NOTE]
> The suggested hyperparameters may change in the future.

| Hyperparameter | $\mathrm{TabM}$ | $\mathrm{TabM_{mini}^\dagger}$ |
| :------------- | :-------------- | :----------------------------- |
| `k`            | 32              | 32                             |
| Depth          | 3               | 2                              |
| Width          | 512             | 512                            |
| Dropout        | 0.1             | TODO                           |
| Bins           | N/A             | TODO                           |
| Learning rate  | 0.002           | TODO                           |
| Weight decay   | 0.0003          | TODO                           |

**Hyperparameter tuning**

If achieving the highest possible performance is not critical,
then 30-50 iterations of the TPE sampler from Optuna should result in a somewhat reasonable configuration.
It the paper:
- For MLP, 100 iterations were used.
- For TabM, 100 iterations were used on smaller datasets, and 50 iterations on larger datasets.

## Metrics

The published results allow easily summarizing the metrics of the models on all datasets.

<details>
<summary>Show how</summary>

```python
import json
from pathlib import Path

import pandas as pd

def get_dataset_name(dataset_path: str) -> str:
    """
    >>> get_dataset_name('data/california')
    'california'
    >>> get_dataset_name('data/regression-num-large-0-year')
    'year'
    """
    name = dataset_path.removeprefix('data/')
    return (
        name.split('-', 4)[-1]  # The "why" benchmark.
        if name.startswith(('classif-', 'regression-'))
        else name
    )

model = 'tabm'  # Or any other model from the exp/ directory.

# Load all training runs.
df = pd.json_normalize([
    json.loads(x.read_text())
    for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
])
df['Dataset'] = df['config.data.path'].map(get_dataset_name)

# Aggregate the results over the random seeds.
print(df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std']))
```

The output exactly matches the metrics reported in the very last section of the paper:

```
                                             mean         std
Dataset                                                      
Ailerons                                -0.000157    0.000002
Bike_Sharing_Demand                    -42.108096    0.501597
Brazilian_houses                        -0.044310    0.021299
KDDCup09_upselling                       0.800227    0.010331
MagicTelescope                           0.860680    0.005765
Mercedes_Benz_Greener_Manufacturing     -8.221496    0.894050
MiamiHousing2016                        -0.148294    0.003001
MiniBooNE                                0.950001    0.000545
OnlineNewsPopularity                    -0.858395    0.000325
SGEMM_GPU_kernel_performance            -0.015809    0.000385
adult                                    0.858158    0.001100
analcatdata_supreme                     -0.077736    0.009874
bank-marketing                           0.790799    0.006795
black-friday                            -0.687502    0.001464
california                              -0.450932    0.003154
churn                                    0.861300    0.002463
cooking-time                            -0.480330    0.000587
covtype2                                 0.971188    0.000800
cpu_act                                 -2.193951    0.052341
credit                                   0.775121    0.004241
delivery-eta                            -0.550962    0.001511
diamond                                 -0.134209    0.001725
ecom-offers                              0.594809    0.000557
elevators                               -0.001853    0.000025
fifa                                    -0.797377    0.014414
higgs-small                              0.738256    0.002775
homecredit-default                       0.858349    0.001019
homesite-insurance                       0.964121    0.000401
house                               -30002.387181  181.962989
house_sales                             -0.169186    0.001056
isolet                                  -1.883108    0.119444
jannis                                   0.806610    0.001525
kdd_ipums_la_97-small                    0.884546    0.006317
maps-routing                            -0.161169    0.000120
medical_charges                         -0.081265    0.000052
microsoft                               -0.743353    0.000265
nyc-taxi-green-dec-2016                 -0.386578    0.000596
otto                                     0.826756    0.001436
particulate-matter-ukair-2017           -0.368573    0.000628
phoneme                                  0.870065    0.016701
pol                                     -3.359482    0.401706
road-safety                              0.794583    0.001253
sberbank-housing                        -0.246943    0.003539
sulfur                                  -0.019162    0.003538
superconduct                           -10.337929    0.033769
visualizing_soil                        -0.124183    0.018830
weather                                 -1.478620    0.003926
wine                                     0.796127    0.013558
wine_quality                            -0.616949    0.012259
year                                    -8.870127    0.011037
```

</details>

# Set up the environment

## Software

Clone the repository:

```shell
git clone https://github.com/yandex-research/tabm
cd tabm
```

Install any of the following tools, and follow the remaining instructions.

| Tool                                                                                           | Supported OS | Supported devices |
| :--------------------------------------------------------------------------------------------- | :----------- | :---------------- |
| [Pixi](https://pixi.sh/latest/#installation)                                                   | Linux, macOS | CPU, GPU          |
| [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) | Linux        | GPU               |
| [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)           | Linux        | GPU               |
| [Conda](https://conda-forge.org/download/)                                                     | Linux        | GPU               |

<details>
<summary>What is Pixi?</summary>

Pixi is a high-level tool built on top of Conda environments.

*The main benefits:*

- **Pixi allows running the code seamlessly across different operating systems, with and without GPUs**
  (this projects supports only Linux and macOS).
- **Pixi guarantees that you and your collaborators will always have exactly the same Conda environment.**
  If anyone adds/removes/updates a package, the enviroment is updated for all collaborators through the `pixi.lock` file.
  In particular, you will have exactly the same environment as us, the original authors.

*Technical details:*

- **Pixi is a single binary file that can be downloaded and run as-is.**
  Thus, Pixi will not affect your current installations of pip, uv, conda, and of any other tools.
- **Pixi will automatically create and manage Conda environments in the `.pixi` folder right in the root of this repository.** No need for manually creating the environment,
  installing packages, etc. At any moment, it is totally safe to remove the `.pixi` directory from the root of this repository.
- **All Pixi commands must be run from the root of the repository.**

For more details, see the official Pixi documentation.

</details>

With Pixi, the environment will be created automatically
when you do `pixi run` or `pixi shell` for the first time. For example, try:

```shell
# Running commands in the environment with GPU
pixi run -e cuda python -c "import torch; print(torch.cuda.is_available())"

# Running commands in the CPU-only environment
pixi run python --version
```

Or, if you prefer the Conda style workflow:

```shell
# Activate the environment with GPU
pixi shell -e cuda

# Activate the CPU-only environment
pixi shell

# Running commands (without `pixi run`)
python --version

# Deactivate the environment
exit
```

With Micromamba:

```
micromamba create -f environment.yaml
micromamba activate tabm
```

With Mamba:

```
mamba create -f environment.yaml
mamba activate tabm
```

With Conda:

```
conda create -f environment.yaml
conda activate tabm
```

## Data

***License:** we do not impose any new license restrictions in addition to the original licenses of the used dataset. See the paper to learn about the dataset sources.*

The data consists of two parts.

**Part 1.** Go to the root of the repository and run:

```
mkdir local
wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar -O local/tabm-data.tar.gz
mkdir data
tar -xvf local/tabm-data.tar.gz -C data
```

**Part 2.** Create the `local` directory
and download the [TabReD](https://github.com/yandex-research/tabred) benchmark to `local/tabred`
(you will need an account on Kaggle).
Then, run:

```
python tools/prepare_tabred.py local/tabred data
```

## Quick test

To check that the environment is configured correctly,
run the following command and wait for the training to finish.
Please, note:
- The first run in a newly created environment can be (very) slow to start.
- The results of the experiment will not be representative.
  It is needed only to test the environment.

```shell
# Pixi with GPU
pixi run -e cuda python bin/model.py exp/debug/0.toml --force

# Pixi without GPU
pixi run python bin/model.py exp/debug/0.toml --force

# Without Pixi
python bin/model.py exp/debug/0.toml --force
```

The last line of the output log should look like this:
```
[<<<] exp/debug/0 | <date & time>
```

# Running the code

This section will be useful if you are planning any of the following:

- Reproducing the results from the paper.
- Tuning and training the models on custom datasets.
- Using this repository as a starting point for future work.

## Code overview

| Code              | Comment                                                   |
| :---------------- | :-------------------------------------------------------- |
| `bin/model.py`    | **The implementation of TabM** and the training pipeline  |
| `bin/tune.py`     | Hyperparameter tuning                                     |
| `bin/evaluate.py` | Evaluating a model under multiple random seeds            |
| `bin/ensemble.py` | Evaluate an ensemble of models                            |
| `bin/go.py`       | `bin/tune.py` + `bin/evaluate.py` + `bin/ensemble.py`     |
| `lib`             | Common utilities used by the scripts in `bin`             |
| `exp`             | Hyperparameters and metrics of the models on all datasets |
| `tools`           | Additional technical tools                                |

The `exp` directory is structured as follows:

```
exp/
  <model>/
    <dataset>/       # Or why/<dataset> or tabred/<dataset>
      0-tuning.toml  # The hyperparameter tuning config
      0-tuning/      # The result of the hyperparameter tuning
      0-evaluation/  # The evaluation under multiple random seeds
```

**Models**

All available models are represented by the `Model` class from `bin/model.py`.
They differ in the `arch_type`, `k`, `num_embeddings` and `bins` arguments.
The values for these arguments can be inferred from the TOML configs in the `exp` directory,
and from the `bin/model.py` script, where `Model` is used.

The following table is the mapping between the models and their subdirectories in `exp`.

| Model                          | Experiments                     |
| :----------------------------- | :------------------------------ |
| $\mathrm{MLP}$                 | `exp/mlp`                       |
| $\mathrm{MLP^\dagger}$         | `exp/mlp-piecewiselinear`       |
| $\mathrm{MLP{-}PLR}$           | `exp/mlp-periodic`              |
| $\mathrm{MLP{-}PLR(lite)}$     | `exp/mlp-periodiclite`          |
| $\mathrm{TabM_{mini}}$         | `exp/tabm-mini`                 |
| $\mathrm{TabM}$                | `exp/tabm`                      |
| $\mathrm{TabM_{mini}^\dagger}$ | `exp/tabm-mini-piecewiselinear` |
| $\mathrm{TabM^\dagger}$        | `exp/tabm-piecewiselinear`      |

## Common guidelines

On your first reading, feel free to skip this section.

<details>
<summary>Show</summary>

- `bin/model.py` takes one TOML config as the input and produces a directory next to the config as the output.
  For example, the command `python bin/model.py exp/hello/world.toml` will produce the directory `exp/hello/world`.
  The `report.json` file in the output directory is the main result of the run:
  it contains all metrics and hyperparameters.
- The same applies to `bin/tune.py`.
- Some scripts support the `--continue` flag to continue the execution of an interrupted run.
- Some scripts support the `--force` flag to **overwrite the existing result**
  and run the script from scratch.
- The layout in the `exp` directory can be arbitrary;
  the current layout is just our convention.

</details>

## Training

To train a model once, compose a TOML config with hyperparameters and pass it to `bin/model.py`.
For example, the following command reproduces one training run of TabM on the California Housing dataset:

```
mkdir -p exp/reproduce/train-once
cp exp/tabm/california/0-evaluation/0.toml exp/reproduce/train-once/
python bin/model.py exp/reproduce/train-once/0.toml
```

The output will be located in the `0` directory next to the TOML config.

## Hyperparameter tuning

Use `bin/tune.py` to tune hyperparameters for `bin/model.py`.
For example, the following commands reproduce the hyperparameter runing of TabM on the California Housing dataset
(this takes around one hour on NVIDIA A100):

```
mkdir -p exp/reproduce/tabm/california
cp exp/tabm/california/0-tuning.toml exp/reproduce/tabm/california
python bin/tune.py exp/reproduce/tabm/california/0-tuning.toml --continue
```

## Evaluation

Use `bin/evaluate.py` to train a model under multiple random seeds.
For example, the following command evaluates the tuned TabM from the previous section:

```
python bin/evaluate.py exp/reproduce/tabm/california/0-tuning
```

To evaluate a manually composed config for `bin/model.py`,
create a directory with a name ending with `-evaluation`,
and put the config with the name `0.toml` in it.
Then, pass the directory as the argument to `bin/evaluate.py`.
For example:

```
# The config is stored at exp/<any/path>/0-evaluation/0.toml
python bin/evaluate.py exp/<any/path>/0-evaluation
```

## Ensembling

Use `bin/ensemble.py` to compute metrics for an ensemble of *already trained* models.
For example, the following command evaluates an ensemble of the evaluated TabM from the previous section:

```
python bin/ensemble.py exp/reproduce/tabm/california/0-evaluation
```

## Automating all of the above

Use `bin/go.py` to run hyperparameter tuning, evaluation and ensembling with a single command.
For example, all the above steps can be implemented as follows:

```
mkdir -p exp/reproduce/tabm-go/california
cp exp/tabm/california/0-tuning.toml exp/reproduce/tabm-go/california

python bin/go.py exp/reproduce/tabm-go/california/0-tuning --continue
```

# Adding new datasets

*New datasets must follow the layout and NumPy data types of the datasets in `data/`.*

Let's assume your dataset is called `my-dataset`.
Then, create the `data/my-dataset` directory with the following layout:

```
data/
  my-dataset/
    # Continuous features, if presented
    # NumPy data type: np.float32
    X_num_train.npy
    X_num_val.npy
    X_num_test.npy

    # Categorical features, if presented
    # NumPy data type: np.str_ (i.e. string)
    X_cat_train.npy
    X_cat_val.npy
    X_cat_test.npy

    # Binary features, if presented
    # NumPy data type: np.float32
    X_bin_train.npy
    X_bin_val.npy
    X_bin_test.npy

    # Labels
    # NumPy data type (regression): np.float32
    # NumPy data type (classification): np.int64
    Y_train.npy
    Y_val.npy
    Y_test.npy

    # Dataset information in the JSON format:
    # {
    #     (required) "task_type": < "regression" or "binclass" or "multiclass"     >,
    #     (optional) "name":      < The full dataset name, e.g. "My Dataset"       >,
    #     (optional) "id":        < Any string unique across all datasets in data/ >
    # 
    # }
    info.json

    # Just an empty file
    READY
```

# How to cite

```
@article{gorishniy2024tabm,
    title={{TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling}},
    author={Yury Gorishniy and Akim Kotelnikov and Artem Babenko},
    journal={{arXiv}},
    volume={2410.24210},
    year={2024},
}
```
