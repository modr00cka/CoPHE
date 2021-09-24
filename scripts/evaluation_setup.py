import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

def load_translation_csv(transition_csv_file='../ICD9/icd9_transitions.csv'):
    """
    Load the graph translation csv (product of graph_setup.py)
    """   
    translation_dict_icd9 = pd.read_csv(transition_csv_file)
    return translation_dict_icd9

def set_up_translation_matrix(graph_df, source_id_dict, source_label_list, level):
    relevant_graph_df = graph_df[graph_df["level"]==level]
    relevant_graph_df = relevant_graph_df[relevant_graph_df["source"].isin(source_label_list)]
    sources = relevant_graph_df.source
    targets = relevant_graph_df.target
    print(list(sources), list(targets))
    target_set = list(set(targets))
    target_id_dict = dict(zip(target_set, range(len(target_set))))
    rows = [source_id_dict[code] for code in sources]
    cols = [target_id_dict[code] for code in targets]
    extra_sources = list(set(source_label_list).difference(set(sources)))
    extra_rows = [source_id_dict[code] for code in extra_sources]
    extra_cols = [0]*len(extra_rows)
    extra_vals = [0]*len(extra_rows)
    vals = [1]*len(rows)+extra_vals
    rows = rows+extra_rows
    cols = cols+extra_cols
    print(rows, cols, vals)
    trans_matrix = csr_matrix((vals,(rows, cols)), shape = (len(source_label_list), len(cols)))
    print(trans_matrix.toarray())
    return trans_matrix, target_id_dict
    
def setup_matrices_by_layer(code_ids, translation_df, source_label_list, max_layer = 1):
    matrices = []
    target_id_dicts = []
    for level in range(max_layer):
        matrix, id_dict = set_up_translation_matrix(translation_df, code_ids, source_label_list, level)
        matrices.append(matrix)
        target_id_dicts.append(id_dict)
    return matrices, target_id_dicts
    
def combined_matrix_setup(code_ids, translation_dict, source_label_list, max_layer = 1):
    return (setup_matrices_by_layer(code_ids, translation_dict, source_label_list, max_layer))
    # matrices, level_id_dicts = setup_matrices_by_layer(code_ids, translation_dict, max_layer, include_duplicates)
    # return [low_level_matrix]+matrices, [low_level_id_dict]+level_id_dicts
                
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
    for i in range(max_onto_layers+1):
        translation_matrix = layer_matrices[i] # layer matrix retrieval
        translated_preds, translated_golds = preds*translation_matrix, golds*translation_matrix # translation from flat predictions into the layer
        combined_preds.append(translated_preds)
        combined_golds.append(translated_golds)
    
    # concatenation between layers for predictions and true labels respectively
    combined_preds = np.concatenate(combined_preds, 1)
    combined_golds = np.concatenate(combined_golds, 1)
    
    return combined_preds, combined_golds
    
if __name__ == "__main__":
    print(f"Hierarchical Evaluation Setup Demonstration")
    print(f"Vectors correspond to leafs: \n(a.1, a.2, a.3, b.1, b.2, c.1, d)")
    print(f"Their corresponding layer 1 versions are: \b (a, a, a, b, b, c, d)")
    
    prediction_code_list = ["a.1", "a.2", "a.3", "b.1", "b.2", "c.1", "d"]#"a", "b", "c", , "AB", "CD", "@"]
    full_code_list = ["a.1", "a.2", "a.3", "b.1", "b.2", "c.1", "d", "a", "b", "c", "AB", "CD", "@"]
    
    code_ids = dict(zip(full_code_list, range(len(full_code_list))))
    translation_list = [("a.2",0,"a.2"), ("a.2",1,"a"), ("a.2",2,"AB"), ("a.2",3,"@"),
                        ("a.1",0,"a.1"), ("a.1",1,"a"), ("a.1",2,"AB"), ("a.1",3,"@"),
                        ("a.3",0,"a.3"), ("a.3",1,"a"), ("a.3",2,"AB"), ("a.3",3,"@"),
                        ("b.1",0,"b.1"), ("b.1",1,"b"), ("b.1",2,"AB"), ("b.1",3,"@"),
                        ("b.2",0,"b.2"), ("b.2",1,"b"), ("b.2",2,"AB"), ("b.2",3,"@"),
                        ("c.1",0,"c.1"), ("c.1",1,"c"), ("c.1",2,"CD"), ("c.1",3,"@"),
                        ("a",1,"a"), ("a",2,"AB"), ("a",3,"@"),
                        ("b",1,"b"), ("b",2,"AB"), ("b",3,"@"),
                        ("c",1,"c"), ("c",2,"CD"), ("c",3,"@"),
                        ("d",1,"d"), ("d",2,"CD"), ("d",3,"@"),
                        ("AB",2,"AB"), ("AB",3,"@"),
                        ("CD",2,"CD"), ("CD",3,"@"),
                        ("@",3,"@")]
    translation_df = pd.DataFrame(translation_list, columns=['source', 'level', 'target'])
                             
                             
    matrices, layer_id_dicts  = (combined_matrix_setup(code_ids, translation_df, prediction_code_list, max_layer = 3))
    print("========TRANSLATION MATRICES========")
    print("Leaves to Layer 0")
    print(matrices[0].toarray(), layer_id_dicts[0])
    print("====================================")
    print("Leaves to 1")
    print(matrices[1].toarray(), layer_id_dicts[1])
    print("====================================")
    print("Leaves to 2")
    print(matrices[2].toarray(), layer_id_dicts[2])


    sample_matrix = np.array([[0, 1, 1, 0, 1, 0, 0],  
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 1, 1, 0, 0, 1],
                              [0, 0, 1, 1, 1, 0, 0],
                              [1, 1, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1]])
                              
    print("========Sample Transitions========")
    print("Sample prediction Matrix:")
    print(sample_matrix)
    print("Layer 0")
    print((sample_matrix.dot(matrices[0].toarray())), layer_id_dicts[0])
    print("Layer 1")
    print((sample_matrix.dot(matrices[1].toarray())), layer_id_dicts[1])
    print("Layer 2")
    print((sample_matrix.dot(matrices[2].toarray())), layer_id_dicts[2])
    
    
    print("Sample gold standard Matrix:")
    sample_gold_matrix = np.array([[0, 0, 1, 0, 1, 0, 1],
                                   [0, 1, 0, 0, 0, 1, 0],
                                   [1, 0, 1, 1, 0, 1, 0],
                                   [0, 0, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 1],
                                   [0, 0, 1, 1, 1, 0, 1]])
    print(sample_gold_matrix)
    print("========Overall Cross-Layer Evaluation Setup========")
    
    combined_preds, combined_golds = hierarchical_eval_setup(sample_matrix, sample_gold_matrix, matrices, 2)
    print("Combined prediction vectors across layers")
    print(combined_preds)
    print("Combined gold standard vectors across layers")
    print(combined_golds)
    print("With these combined predictions and gold standard labels across layers we can now apply the evaluation measures for the non-binary scenario in multi_level_eval.py")
    
    #another example: about ICD9 graph
    print("The ICD9 graph example")
    #load json to get the  translation_dict from icd-9
    fn_icd9_graph_json = r"/afs/inf.ed.ac.uk/user/s12/s1206296/Desktop/CDT/PhD/PhD/code/repos/multi_level_eval_fresh/icd9_hierarchical_eval/EUROVOC/eurovoc_en.csv"
    translation_dict_icd9 = load_translation_csv(fn_icd9_graph_json)
    
    print("There are %d entries in translation_dict_icd9." % len(translation_dict_icd9))
    
    code_ids = dict(zip(["4053", "5403", "199"], range(3)))
    matrices, layer_id_dicts  = (setup_matrices_by_layer(code_ids, translation_dict_icd9, ["4053", "5403", "199"], max_layer = 5))
    print("========TRANSLATION MATRICES========")
    for level in range(1,7):
        print(f"Leave to {level}")
        print(matrices[level-1].toarray(), layer_id_dicts[level-1])
        print("====================================")