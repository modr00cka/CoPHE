# CoPHE

Count-preserving hierarchical evaluation and set-based hierarchical evaluation methods for hierarchical label spaces.
Currently implemented only for the label space of the ICD-9 ontology of diseases and procedures.

## Requirements

``numpy``
``pandas``
``scipy.sparse.csr_matrix``


## Scripts

### evaluation_setup.py
This script cotains methods for setting up the evaluation.

``setup_matrices_by_layer.py`` produces transition matrices between the original prediction and each layer, along with the resulting ancestor code IDs within the vectors created through multiplication by transition matrices.


``hierarchical_eval_setup.py`` concatenates the predictions and gold standard across layers respectively. This results in overall predictions (with ancestors) and overall gold standard (with ancestors). These can then be evaluated with methods from ``multi_level_eval.py``

### mutli_level_eval.py 
This script includes the evaluation measures - either overall, or per class; binary and non-binary. It also includes reporting functions for precision, recall, and F1. The ``report`` method produces these for each class and presents them as a dataframe.

The intended use is to create individual reports for each of the layers for in-depth analysis, and to run an overall micro-average report on the concatenated matrices received from ``hierarchical_eval_setup`` from ``evaluation_setup.py``

## Theoretical background

For a theoretical background behind this implementation, please refer to [link_to_publication].
An informal summary can be found in ``Summary.md``.