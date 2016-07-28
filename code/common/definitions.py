import numpy as np

def __read_dep_graph(string):
    parts = string.strip().split(' ')
    return np.array([float(x) for x in parts])

def __write_dep_graph(graph):
    return ' '.join([str(x) for x in graph])

'''
Defines the stanford and universal POS-tag sets:
'''

stanford_pos=['TO', 'VBG', 'HYPH', 'VBP', "\'\'", 'JJR', 'NNPS', 'AFX', 'POS', ',', 'VB', 'RBR', 'WDT', 'WRB', 'RBS', 'NFP', 'WP', 'WP$', 'XX', 'MD', '-RRB-', 'CC', 'FW', 'JJ', 'VBZ', 'GW', 'VBD', 'UH', '-LRB-', 'RP', 'LS', '.', '``', 'ADD', 'NNP', '$', 'JJS', 'IN', ':', 'CD', 'NNS', 'VBN', 'DT', 'PDT', 'EX', 'PRP$', 'SYM', 'RB', 'PRP', 'NN']

universal_pos=['NUM', 'DET', 'PUNCT', 'ADJ', 'X', 'SYM', 'AUX', 'NOUN', 'VERB', 'PRON', 'ADV', 'ADP', 'CONJ', 'INTJ', 'SCONJ', 'PROPN', 'PART']

'''
Defines the conll sentence file structure. 
Terminology is (header name, function to read element, function to write element).
'''
conll_column_headers = [('id', int, str),
                        ('token', str, str),
                        ('lemma', str, str),
                        ('universal_pos', str, str),
                        ('stanford_pos', str, str),
                        ('features', str, str),
                        ('dependency_head_id', int, str),
                        ('dependency_head_relation', str, str),
                        ('weird_underscore', str, str),
                        ('space_after', str, str),
                        ('dependency_graph', __read_dep_graph, __write_dep_graph)]
