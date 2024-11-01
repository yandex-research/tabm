# TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling<!-- omit in toc -->

<!-- :scroll: [arXiv](TODO(link)) -->
&nbsp; :books: [RTDL (other projects on tabular DL)](https://github.com/yandex-research/rtdl)

*TL;DR: TabM is a simple and powerful tabular DL architecture that efficiently imitates an ensemble of MLPs.*

<!-- The official implementation of the paper
"TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling". -->

> [!IMPORTANT]
> For a quick overview, see **the abstract, Figure 1 and Page 7** in the paper.

---

Table of contents
- [Overview](#overview)
  - [Code](#code)
  - [Metrics](#metrics)
- [Set up the environment](#set-up-the-environment)
  - [Software](#software)
  - [Data](#data)
  - [Test](#test)
- [Using the repository](#using-the-repository)
  - [Common guidelines](#common-guidelines)
  - [Training](#training)
  - [Hyperparameter tuning](#hyperparameter-tuning)
  - [Evaluation](#evaluation)
  - [Ensembling](#ensembling)
  - [Automating all of the above](#automating-all-of-the-above)
- [Adding new datasets and metrics](#adding-new-datasets-and-metrics)
- [How to cite](#how-to-cite)

---

# Overview

This section provides a brief overview of the project.
Running the code, including training and hyperparameter tuning, is covered later in this document.

## Code

| Code              | Comment                                                  |
| :---------------- | :------------------------------------------------------- |
| `bin/model.py`    | **The implementation of TabM** and the training pipeline |
| `bin/tune.py`     | Hyperparameter tuning                                    |
| `bin/evaluate.py` | Evaluating a model under multiple random seeds           |
| `bin/ensemble.py` | Evaluate an ensemble of models                           |
| `bin/go.py`       | `bin/tune.py` + `bin/evaluate.py` + `bin/evaluate.py`    |
| `lib`             | Common utilities used by the scripts in `bin`            |
| `exp`             | Hyperparameters and metrics on all datasets              |
| `tools`           | Additional technical tools                               |

## Metrics

The `exp/` directory allows easily exploring the metrics of models on all datasets.
For example, this is how to summarize the metrics for TabM:

<details>
<summary>Code</summary>

<!-- > [!TIP]
> With [Pixi](https://pixi.sh), you can simply do `pixi run jupyter-lab` and run the code below as-is: -->

```python
import json
from pathlib import Path

import pandas as pd

def get_dataset_name(dataset_path: str) -> str:
    name = dataset_path.removeprefix('data/')
    return (
        name.split('-', 4)[-1]  # The "why" benchmark.
        if name.startswith(('classif-', 'regression-'))
        else name
    )

df = pd.json_normalize([
    json.loads(x.read_text())
    for x in Path('exp').glob('tabm/**/0-evaluation/*/report.json')
])
df['Dataset'] = df['config.data.path'].map(get_dataset_name)

print(df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std']))
```

</details>

The output exactly matches the metrics reported in the very last section of the paper:

<details>
<summary>Output</summary>

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

**Requirements**
- OS: Linux or macOS.
- Hardware: GPU is not required, but highly recommended for running compute-heavy pipelines.

**Step 1.** Clone the repository:

```shell
git clone https://github.com/yandex-research/tabm
cd tabm
```

**Step 2.**
To manage the project, use any of the following tools:
[Pixi](https://github.com/prefix-dev/pixi/?tab=readme-ov-file#installation) (CPU or GPU),
[Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (requires GPU),
[Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)  (requires GPU),
[Conda](https://conda-forge.org/download/) (requires GPU).

With Pixi, the environment will be created automatically when you run any command. For example, try:

```
# On a CPU-only machine
pixi run python --version

# On a machine with GPU
pixi run -e cuda python -m "import torch; print(torch.cuda.is_available())"
```

<details>
<summary><i>If you are new to Pixi</i></summary>

The main benefits of Pixi for this project:

1. Pixi allows seamlessly running the code on Linux and macOS, with and without GPUs.
   For example, if you are reading this from a laptop,
   then your machine is one command away from running any code in this repository.
2. Pixi guarantees that you and all your collaborators will always have exactly the same environment.
   If anyone adds/removes a package, the enviroment is updated for all collaborators.
   In particular, you will have exactly the same environment as us, the original authors.

**A three-minute crash course**

- Pixi is a single binary file that can be downloaded and run as-is.
  Thus, Pixi will not affect your current installations of pip, uv, conda, and of any other tools.
- Pixi will automatically create and manage conda environments in the `.pixi` folder
  right in the root of this repository. No need for manually creating the environment,
  installing packages, etc.
- All Pixi commands must be run from the root of the repository.
- At any moment, it is totally safe to remove the `.pixi` directory from the root of this repository.

There are two ways to run commands:

- The Pixi way: `pixi run <command>`.
  Pixi will automatically validate and activate the conda environment before running your command.
  If the environment has not been created yet, Pixi will automatically create it.
- The Conda way: activate the environment with `pixi shell` and run `<command>` as-is.

*The examples below cover both approaches.*

Run any command in the default conda environment without GPU:

```
pixi run <any command with any arguments>

# Example
pixi run python --version
```

Run any command in the conda environment with GPU:

```
pixi run -e cuda <any command with any arguments>

# Example
pixi run -e cuda python -c "import torch; print(torch.cuda.is_available)"
```

Activate the default environment without GPU:

```
pixi shell
```

Activate the environment with GPU:

```
pixi shell -e cuda
```

Once the environment is activated, run any commands, as you would do with Conda.

To deactivate the current environment:

```
exit
```

The above commands are enough to run the code in this repository.
For more details, see the official Pixi documentation.

</details>

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

**Step 3.**
To run any `<command>` with GPU, the `CUDA_VISIBLE_DEVICES` environment variable must be explicitly set:

```
# Like this
export CUDA_VISIBLE_DEVICES="0"
<command>

# Or like this
CUDA_VISIBLE_DEVICES="0" <command>
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

After that, the `data/` directory should appear.

**Part 2.** Create the `local` directory
and download the [TabReD](https://github.com/yandex-research/tabred) benchmark to `local/tabred`
(you will need an account on Kaggle).
Then, run:

```
python tools/prepare_tabred.py local/tabred data
```

## Test

To check that the environment is configured correctly,
run the following command and wait for the training to finish.
Please, note:
- The first run in a newly created environment can be slow to start.
- The results of the experiment will not be representative.
  It is needed only to test the environment.

```
export CUDA_VISIBLE_DEVICES=0

# Pixi without GPU
pixi run python bin/model.py exp/debug/0.toml --force

# Pixi with GPU
pixi run -e cuda python bin/model.py exp/debug/0.toml --force

# Without Pixi
python bin/model.py exp/debug/0.toml --force
```

The last line of the output log should look like this:
```
[<<<] exp/debug/0 | <date & time>
```

# Using the repository

This section is useful for:

- Tuning and training the models on custom datasets.
- Reproducing the results from the paper.
- Using this repository as a starting point for future work.

## Common guidelines

During the first read, feel free to skip this section.

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
- The layout in the `exp` directory can be arbitrary.
  The `exp/<model>/<dataset>/<activity>/<results>` pattern is just our convention.

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
python bin/evaluate.py exp/reproduce/tabm/california/0-tuning --continue
```

To evaluate a manually composed config for `bin/model.py`,
it must be named `0.toml` and the name of its parent directory must end with `-evaluation`.
For example:

```
python bin/evaluate.py exp/<some/path/hello/world>/0-evaluation/0.toml --continue
```

## Ensembling

Use `bin/ensemble.py` to compute metrics for an ensemble of *already trained* models.
For example, the following command evaluates an ensemble of the evaluated TabM from the previous section:

```
python bin/evaluate.py exp/reproduce/tabm/california/0-tuning
```

## Automating all of the above

Use `bin/go.py` to run hyperparameter tuning, evaluation and ensembling in one command.
For example, all the above steps can be implemented in one command:

```
# Create
mkdir -p exp/reproduce/tabm-go/california
cp exp/tabm/california/0-tuning.toml exp/reproduce2/tabm-go/california

python bin/go.py exp/reproduce/tabm-go/california/0-tuning --continue
```

# Adding new datasets and metrics

New datasets must follow the layout and NumPy data types of the datasets in `data/`.
A good example is the `data/adult` dataset, because it contains all types of features.

Let's assume your dataset is called `my-dataset`.
Then, create the `data/my-dataset` directory with the following content:

1. If the dataset has continuous (a.k.a. "numerical") features
    - Files: `X_num_train.npy`, `X_num_val.npy`, `X_num_test.npy`
    - NumPy data type: `np.float32`
2. If the dataset has binary features
    - Files: `X_bin_train.npy`, `X_bin_val.npy`, `X_bin_test.npy`
    - NumPy data type: `np.float32`
    - All values must be `0.0` and `1.0`
3. If the dataset has categorical features
    - Files: `X_cat_train.npy`, `X_cat_val.npy`, `X_cat_test.npy`
    - NumPy data type: `np.str_` (**yes, the values must be strings**)
4. Labels
    - Files: `Y_train.npy`, `Y_val.npy`, `Y_test.npy`
    - NumPy data type: `np.float32` for regression, `np.int64` for classification
    - For classification problems, the labels must form the range `[0, ..., n_classes - 1]`.
5. `info.json` -- a JSON file with the following keys:
    - `"task_type"`: one of `"regression"`, `"binclass"`, `"multiclass"`
    - (optional) `"name"`: any string (a "pretty" name for your dataset, e.g. `"My Dataset"`)
    - (optional) `"id"`: any string (must be unique among all `"id"` keys of all `info.json` files of all datasets in `data/`)
6. `READY` -- just an empty file

# How to cite

```
@article{gorishniy2024tabm,
    title={TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling},
    author={Yury Gorishniy and Akim Kotelnikov and Artem Babenko},
    journal={{arXiv}},
    volume={TODO},
    year={2024},
}
```
