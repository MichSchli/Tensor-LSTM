import imp
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Parse a file using a stored model.")
parser.add_argument("--features", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--sentences", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--dev_features", help="Development feature filepath (CoNLL format).", required=True)
parser.add_argument("--dev_sentences", help="Development sentence filepath (CoNLL format).", required=True)
parser.add_argument("--model_path", help="Model path.", required=True)
parser.add_argument("--algorithm", help="Chosen algorithm.", required=True)
parser.add_argument("--feature_mode", help="Feature set in use.", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')
algorithm = imp.load_source('io', 'code/parsing/algorithms/'+args.algorithm+'.py')

features = {}
dev_features = {}

if args.feature_mode == 'sentence':
    features['sentence'] = io.read_sentence_features(args.features+'.sentence.feat')
elif args.feature_mode == 'character':
    features['character'] = io.read_character_features(args.features+'.character.feat')
elif args.feature_mode == 'both':
    features['sentence'] = io.read_sentence_features(args.features+'.sentence.feat')
    features['character'] = io.read_character_features(args.features+'.character.feat')

sentences = io.read_conll_sentences(args.sentences)
labels = [[token['dependency_graph'] for token in sentence] for sentence in sentences]

if args.feature_mode == 'sentence':
    dev_features['sentence'] = io.read_sentence_features(args.dev_features+'.sentence.feat')
elif args.feature_mode == 'character':
    dev_features['character'] = io.read_character_features(args.dev_features+'.character.feat')
elif args.feature_mode == 'both':
    dev_features['sentence'] = io.read_sentence_features(args.dev_features+'.sentence.feat')
    dev_features['character'] = io.read_character_features(args.dev_features+'.character.feat')

dev_sentences = io.read_conll_sentences(args.dev_sentences)
dev_labels = [[token['dependency_graph'] for token in sentence] for sentence in dev_sentences]

algorithm.fit(features, labels, dev_features, dev_labels, model_path=args.model_path)
