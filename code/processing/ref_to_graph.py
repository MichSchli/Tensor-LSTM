import imp
import argparse

io = imp.load_source('io', 'code/common/io.py')

parser = argparse.ArgumentParser(description="Convert a reference-to-head format to a graph-format.")
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (CoNLL format).", required=True)
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)

for sentence in sentences:
    token_count = len(sentence)
    
    for token in sentence:
        head = token['dependency_head_id']
        token['dependency_graph'] = [1.0 if i == head else 0.0 for i in range(token_count + 1)]


io.write_conll_sentences(sentences, args.outfile)
