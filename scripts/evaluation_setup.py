import json
import logging

import numpy as np
from scipy.sparse import csr_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_translation_dict_from_icd9(fn_icd9_graph_json="../ICD9/icd9_graph_desc.json"):
    """
    Load the icd9 graph translation dictionary
    """
    with open(fn_icd9_graph_json, encoding="utf-8") as json_file:
        translation_dict_icd9 = json.load(json_file)
    return translation_dict_icd9


def load_translation_dict_from_icd10(
    fn_icd10_graph_json="../ICD10/icd10_graph_desc.json",
):
    """
    Load the icd10 graph translation dictionary
    """
    with open(fn_icd10_graph_json, encoding="utf-8") as json_file:
        translation_dict_icd10 = json.load(json_file)
    return translation_dict_icd10


def setup_matrices_by_layer(
    code_ids, translation_dict, max_layer=1, include_duplicates=False
):
    """
    Sets up the transition matrices and ID dictionaries for each layer of the ontology up to a maximum value (from the bottom up).
    sample_code_ids - a dictionary mapping IDs in the output layer to codes
    translation_dict - a dictionary containing the codes' ordered parent list (coming from the .json file provided in the ICD9 folder)
    max_layer - integer maximum layer of the ontology (from the bottom up) up to which the hierarchical evaluation is applied
    include_duplicates - boolean, default = True; maintains duplication across lower layers if a leaf is not present in the lowest layer (results in presence of all leafs in all layers)
    returns a tuple:
        matrices - a list of transition matrices from the leaves to each layer of the ontology up to max_layer (from bottom up)
        layer_id_dicts - a list of dictionaries of code IDs in vectors for each layer of the ontology up to max_layer (from the bottom up)
    """
    matrices = []  # tranlsation matrices per layer
    layer_id_dicts = []  # id-to-code dictionary per layer

    for layer in range(max_layer):
        rows, cols, vals = [], [], []  # setup for a sparse matrix
        layer_codeset = (
            set()
        )  # codeset for ancestors - in order to remove duplicates for layer representation
        for code in code_ids:
            candidate = translation_dict[code]["parents"][layer]
            if (
                layer == max_layer - 1
                or candidate != translation_dict[code]["parents"][layer + 1]
                or include_duplicates
            ):
                layer_codeset.add(
                    translation_dict[code]["parents"][layer]
                )  # collection of relevant ancestors in the layer

        layer_ranges = list(range(len(layer_codeset)))
        layer_id_dict = dict(
            zip(list(layer_codeset), layer_ranges)
        )  # association of IDs with relevant acestors in the layer

        for code in code_ids:
            ancestor = translation_dict[code]["parents"][layer]
            if ancestor in layer_codeset:
                rows.append(code_ids[code])  # row number (current code)
                cols.append(layer_id_dict[ancestor])  # col number (ancestor)
                if (
                    include_duplicates or layer == max_layer - 2
                ):  # if duplicates are allowed or the next layer is the final layer, create an edge
                    vals.append(1)
                else:  # otherwise observe the ancestor of the ancestor - if this matches the current code, do not create an edge. Otherwise create an edge.
                    assert (
                        ancestor in translation_dict
                    ), f"Ancestor {ancestor} of code {code} not found."
                    double_ancestor = translation_dict[ancestor]["parents"][layer + 1]
                    duplicate_ancestor = double_ancestor == code
                    vals.append(not (duplicate_ancestor) * 1)

        matrix = csr_matrix(
            (vals, (rows, cols)), shape=((len(code_ids)), (len(set(cols))))
        )  # set up the sparse matrix

        matrices.append(matrix)  # append the matrix for this layer
        layer_id_dicts.append(layer_id_dict)  # append the id dictionary for this layer

    return matrices, layer_id_dicts


def low_level_filter(code_ids, translation_dict):
    """
    Creates the matrix to keep only the lowest-level leaf codes
    """
    rows, cols, vals = [], [], []  # setup for a sparse matrix
    layer_codeset = set()  # set of relevant lowest-level leaves
    for code in code_ids:
        direct_parent = translation_dict[code]["parents"][0]
        if direct_parent != code:
            layer_codeset.add(code)  # collection of relevant lowest-layer codes
    layer_ranges = list(range(len(layer_codeset)))
    layer_id_dict = dict(zip(list(layer_codeset), layer_ranges))
    for code in code_ids:
        rows.append(code_ids[code])  # row number (current code)
        if code in layer_codeset:
            cols.append(layer_id_dict[code])  # col number (ancestor)
            vals.append(1)
        else:
            cols.append(0)
            vals.append(0)
    matrix = csr_matrix((vals, (rows, cols)), shape=((len(rows)), (len(set(cols)))))
    return matrix, layer_id_dict


def combined_matrix_setup(
    code_ids, translation_dict, max_layer=1, include_duplicates=False
):
    low_level_matrix, low_level_id_dict = low_level_filter(code_ids, translation_dict)
    matrices, level_id_dicts = setup_matrices_by_layer(
        code_ids, translation_dict, max_layer, include_duplicates
    )
    return [low_level_matrix] + matrices, [low_level_id_dict] + level_id_dicts


def hierarchical_eval_setup(preds, golds, layer_matrices, max_onto_layers):
    """
    inputs:
      preds - a numpy array, a matrix of predictions
      golds - a numpy array, a matrix of true labels
      layer_matrices - a list of numpy arrays translating the leaf nodes into layers of the ontology
      max_onto_layers - an integer describing the maximum layer (from the bottom up) within the ontology to be evaluated on
    """

    combined_preds = []
    combined_golds = []

    # handling further layers
    for i in range(max_onto_layers + 1):
        translation_matrix = layer_matrices[i]  # layer matrix retrieval
        translated_preds, translated_golds = (
            preds * translation_matrix,
            golds * translation_matrix,
        )  # translation from flat predictions into the layer
        combined_preds.append(translated_preds)
        combined_golds.append(translated_golds)

    # concatenation between layers for predictions and true labels respectively
    combined_preds = np.concatenate(combined_preds, 1)
    combined_golds = np.concatenate(combined_golds, 1)

    return combined_preds, combined_golds


if __name__ == "__main__":
    logging.info(f"Hierarchical Evaluation Setup Demonstration")
    logging.info(f"Vectors correspond to leafs: \n(a.1, a.2, a.3, b.1, b.2, c.1, d)")
    logging.info(f"Their corresponding layer 1 versions are: \b (a, a, a, b, b, c, d)")

    code_list = ["a.1", "a.2", "a.3", "b.1", "b.2", "c.1", "d"]

    code_ids = dict(zip(code_list, range(len(code_list))))
    translation_dict = dict(
        {
            "a.1": dict({"parents": ["a", "AB", "@"]}),
            "a.2": dict({"parents": ["a", "AB", "@"]}),
            "a.3": dict({"parents": ["a", "AB", "@"]}),
            "b.1": dict({"parents": ["b", "AB", "@"]}),
            "b.2": dict({"parents": ["b", "AB", "@"]}),
            "c.1": dict({"parents": ["c", "CD", "@"]}),
            "a": dict({"parents": ["a", "AB", "@"]}),
            "b": dict({"parents": ["b", "AB", "@"]}),
            "c": dict({"parents": ["c", "CD", "@"]}),
            "d": dict({"parents": ["d", "CD", "@"]}),
            "AB": dict({"parents": ["@", "@", "@"]}),
            "CD": dict({"parents": ["@", "@", "@"]}),
        }
    )

    matrices, layer_id_dicts = combined_matrix_setup(
        code_ids, translation_dict, max_layer=2
    )
    logging.info("========TRANSLATION MATRICES========")
    logging.info("Leaves to Layer 0")
    logging.info(matrices[0].toarray(), layer_id_dicts[0])
    logging.info("====================================")
    logging.info("Leaves to 1")
    logging.info(matrices[1].toarray(), layer_id_dicts[1])
    logging.info("====================================")
    logging.info("Leaves to 2")
    logging.info(matrices[2].toarray(), layer_id_dicts[2])

    sample_matrix = np.array(
        [
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )

    logging.info("========Sample Transitions========")
    logging.info("Sample prediction Matrix:")
    logging.info(sample_matrix)
    logging.info("Layer 0")
    logging.info((sample_matrix.dot(matrices[0].toarray())), layer_id_dicts[0])
    logging.info("Layer 1")
    logging.info((sample_matrix.dot(matrices[1].toarray())), layer_id_dicts[1])
    logging.info("Layer 2")
    logging.info((sample_matrix.dot(matrices[2].toarray())), layer_id_dicts[2])

    logging.info("Sample gold standard Matrix:")
    sample_gold_matrix = np.array(
        [
            [0, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 1],
        ]
    )
    logging.info(sample_gold_matrix)
    logging.info("========Overall Cross-Layer Evaluation Setup========")

    combined_preds, combined_golds = hierarchical_eval_setup(
        sample_matrix, sample_gold_matrix, matrices, 2
    )
    logging.info("Combined prediction vectors across layers")
    logging.info(combined_preds)
    logging.info("Combined gold standard vectors across layers")
    logging.info(combined_golds)
    logging.info(
        "With these combined predictions and gold standard labels across layers we can now apply the evaluation measures for the non-binary scenario in multi_level_eval.py"
    )

    # another example: about ICD9 graph
    logging.info("The ICD9 graph example")
    # load json to get the  translation_dict from icd-9
    fn_icd9_graph_json = r"..\ICD9\icd9_graph_desc.json"
    translation_dict_icd9 = load_translation_dict_from_icd9(
        "/disk/scratch/s1206296/PhD/PhD/code/repos/CoPHE-Master/ICD9/icd9_graph_desc.json"
    )

    logging.info(
        "There are %d entries in translation_dict_icd9." % len(translation_dict_icd9)
    )

    code_ids = dict(zip(["770.12", "427.31", "95.25"], range(3)))
    matrices, layer_id_dicts = setup_matrices_by_layer(
        code_ids, translation_dict_icd9, max_layer=2
    )
    logging.info("========TRANSLATION MATRICES========")
    logging.info("Leaves to 1")
    logging.info(matrices[0].toarray(), layer_id_dicts[0])
    logging.info("====================================")
    logging.info("Leaves to 2")
    logging.info(matrices[1].toarray(), layer_id_dicts[1])

    # another example: about ICD9 graph
    logging.info("The ICD10 graph example")
    # load json to get the  translation_dict from icd-9
    fn_icd10_graph_json = r"..\ICD10\icd10_graph_desc.json"
    translation_dict_icd10 = load_translation_dict_from_icd10(
        "/disk/scratch/s1206296/PhD/PhD/code/repos/CoPHE-Master/ICD10/icd10_graph_desc.json"
    )

    logging.info(
        "There are %d entries in translation_dict_icd10." % len(translation_dict_icd10)
    )
    logging.info(translation_dict_icd9["401.9"])
    logging.info(translation_dict_icd10["Q55.63"])
    code_ids = dict(zip(["S52.044Q", "T49.8X5A", "S52.044P"], range(3)))
    matrices, layer_id_dicts = setup_matrices_by_layer(
        code_ids, translation_dict_icd10, max_layer=2
    )
    logging.info("========TRANSLATION MATRICES========")
    logging.info("Leaves to 1")
    logging.info(matrices[0].toarray(), layer_id_dicts[0])
    logging.info("====================================")
    logging.info("Leaves to 2")
    logging.info(matrices[1].toarray(), layer_id_dicts[1])
