import imp
import argparse
import numpy as np

io = imp.load_source('io', 'code/common/io.py')
embeddings = imp.load_source('embedding_reader', 'code/common/embedding_reader.py')
definitions = imp.load_source('definitions', 'code/common/definitions.py')

parser = argparse.ArgumentParser(description="Featurize a conll sentence file using PolyGlot embeddings.")
parser.add_argument("--language", help="Document language.", required=True)
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (Feature format).", required=True)
args = parser.parse_args()

sentence_feature_file = args.outfile + '.sentence.feat'
sentences = io.read_conll_sentences(args.infile)

pos_dict = {tag:idx for idx, tag in enumerate(definitions.universal_pos)}

embedding_model = embeddings.PolyglotReader(args.language)

def __featurize_token(token):
    global embedding_model
    global pos_dict

    feature = embedding_model[token['token'].lower()]

    pos = [0]*len(definitions.universal_pos)
    pos[pos_dict[token['universal_pos']]] = 1
    
    feature = np.concatenate((feature, pos))
    return feature

sentence_features = []

for sentence in sentences:
    sentence_features.append([])
    for token in sentence:
        sentence_features[-1].append(__featurize_token(token))
            
io.write_sentence_features(sentence_features, sentence_feature_file)
