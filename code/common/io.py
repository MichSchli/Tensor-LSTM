import imp
import numpy as np

definitions = imp.load_source('definitions', 'code/common/definitions.py')

conll_column_headers = definitions.conll_column_headers

def conll_list_to_dict(token_list):
    global conll_column_headers

    token_dict = {}
    for i in range(len(token_list)):
        header_name = conll_column_headers[i][0]
        header_type = conll_column_headers[i][1]
        token_dict[header_name] =  header_type(token_list[i])

    return token_dict

def read_raw_conll_sentences(filename):
    
    sentences = [[]]
    
    for line in open(filename, 'r+'):
        stripped = line.strip()

        if stripped:
            token_list = stripped.split('\t')
            sentences[-1].append(token_list)
        else:
            sentences.append([])

    while sentences[-1] == []:
        sentences = sentences[:-1]

    return sentences


def read_conll_sentences(filename):
    
    sentences = [[]]
    
    for line in open(filename, 'r+'):
        stripped = line.strip()

        if stripped:
            token_list = stripped.split('\t')
            token_dict = conll_list_to_dict(token_list)
            sentences[-1].append(token_dict)
        else:
            sentences.append([])

    while sentences[-1] == []:
        sentences = sentences[:-1]

    return sentences


def __write_conll_sentence(sentence, ofile):
    global conll_column_headers

    for token in sentence:                
        formatted_token = [header[2](token[header[0]]) if header[0] in token else '_' for header in conll_column_headers]
        print('\t'.join(formatted_token), file=ofile)

def __write_feature_sentence(sentence, ofile):
    for token in sentence:
        print('\t'.join([str(f) for f in token]), file=ofile)

        
def write_sentence_features(sentences, filename):
    ofile = open(filename, 'w+')

    for sentence in sentences[:-1]:
        __write_feature_sentence(sentence, ofile)
        print('', file=ofile)

    __write_feature_sentence(sentences[-1], ofile)
    ofile.close()

def __write_characters(sentence, ofile):
    for token in sentence:
        print('\t'.join([' '.join([str(f) for f in t]) for t in token]), file=ofile)
    
def write_character_features(sentences, filename):
    ofile = open(filename, 'w+')

    for sentence in sentences[:-1]:
        __write_characters(sentence, ofile)
        print('', file=ofile)

    __write_characters(sentences[-1], ofile)
    ofile.close()


def read_character_features(filename):
    sentences = [[]]
    for line in open(filename, 'r+'):
        stripped = line.strip()

        if stripped:
            token_list = [[float(c) for c in f.split(' ')] for f in stripped.split('\t')]
            feature  = np.array(token_list)
            sentences[-1].append(feature)
        else:
            sentences.append([])

    while sentences[-1] == []:
        sentences = sentences[:-1]

    return sentences    

    
def read_sentence_features(filename):
    sentences = [[]]
    for line in open(filename, 'r+'):
        stripped = line.strip()

        if stripped:
            token_list = [float(f) for f in stripped.split('\t')]
            feature  = np.array(token_list)
            sentences[-1].append(feature)
        else:
            sentences.append([])

    while sentences[-1] == []:
        sentences = sentences[:-1]

    return sentences    


def read_config_file(filename):
    infile = open(filename, 'r+')

    config = {}
    for line in infile:
        if line.strip():
            parts = line.strip().split(':')
            config[parts[0].strip()] = parts[1].strip()

    return config


def write_conll_sentences(sentences, filename):
    ofile = open(filename, 'w+')

    for sentence in sentences[:-1]:
        __write_conll_sentence(sentence, ofile)
        print('', file=ofile)

    __write_conll_sentence(sentences[-1], ofile)
        
    ofile.close()
