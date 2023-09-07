# CoPHE

Count-preserving hierarchical evaluation and set-based hierarchical evaluation methods for hierarchical label spaces.
Currently implemented only for the label space of the ICD-9 ontology of diseases and procedures.

## Requirements

``numpy``
``pandas``
``scipy.sparse.csr_matrix``

The following command will install the necessary dependencies using Conda package manager:
```bash
conda env create -f environment.yaml
conda activate cophe
```

You can also choose not to use this virtual environment should you have the required packages installed.

## Scripts

### evaluation_setup.py
This script cotains methods for setting up the evaluation.

``setup_matrices_by_layer.py`` produces transition matrices between the original prediction and each layer, along with the resulting ancestor code IDs within the vectors created through multiplication by transition matrices.


``hierarchical_eval_setup.py`` concatenates the predictions and gold standard across layers respectively. This results in overall predictions (with ancestors) and overall gold standard (with ancestors). These can then be evaluated with methods from ``multi_level_eval.py``

### multi_level_eval.py 
This script includes the evaluation measures - either overall, or per class; binary and non-binary. It also includes reporting functions for precision, recall, and F1. The ``report`` method produces these for each class and presents them as a dataframe.

The intended use is to create individual reports for each of the layers for in-depth analysis, and to run an overall micro-average report on the concatenated matrices received from ``hierarchical_eval_setup`` from ``evaluation_setup.py``

All scripts are accompanied with test cases to help understand the logic better.
These test cases can be executed by running said scripts:
```bash
python -m scripts.evaluation_setup
# or
python -m scripts.multi_level_eval
```

## Theoretical background

For a theoretical background behind this implementation, please refer to the [CoPHE paper](https://arxiv.org/abs/2109.04853).
An informal summary can be found in ``Summary.md``.