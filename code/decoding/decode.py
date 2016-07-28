import imp
import argparse
import numpy as np
import heapq
import networkx as nx

io = imp.load_source('io', 'code/common/io.py')

parser = argparse.ArgumentParser(description="Convert a reference-to-head formt to a graph-format.")
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (CoNLL format).", required=True)
parser.add_argument("--verbose", help="Print updates after each sentence.", required=False, action="store_true")
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)

'''
Find MSA with Chu-Liu-Edmonds algorithm:
'''
def __find_mst(graph):
    graph = -1*graph
    graph = np.transpose(graph)

    G = nx.DiGraph(graph)
    try:
        MSA = nx.minimum_spanning_arborescence(G)
    except nx.exception.NetworkXException:
        # If a node with no head exists, hack it into a node with uniform dist:
        ng = np.array(graph - 0.0001)
        MSA = nx.minimum_spanning_arborescence(nx.DiGraph(ng))
        
    return nx.adjacency_matrix(MSA).todense().transpose()*(-1)
    
for sentence_idx,sentence in enumerate(sentences):
    if args.verbose:
        print("Decoding sentence "+str(sentence_idx)+"...", end='\r')
    deplist = [token['dependency_graph'] for token in sentence]

    sentence_graph = np.vstack(deplist)
    root = np.zeros((len(sentence)+1))
    sentence_graph = np.vstack((root, sentence_graph))
    
    mst = __find_mst(sentence_graph)

    edges = np.argmax(mst, axis=1)

    for i, token in enumerate(sentence):
        token['dependency_head_id'] = int(edges[i+1])
        token['dependency_graph'] = np.zeros((len(sentence)+1))
        token['dependency_graph'][int(edges[i+1])] = 1.0

io.write_conll_sentences(sentences, args.outfile)

