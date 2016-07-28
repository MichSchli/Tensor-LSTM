import imp
import argparse

parser = argparse.ArgumentParser(description="Parse a file using a stored model.")
parser.add_argument("--gold", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--prediction", help="Input filepath (CoNLL format).", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')

gold = io.read_conll_sentences(args.gold)
prediction = io.read_conll_sentences(args.prediction)

def unlabelled_attachment_score(gold_sentences, pred_sentences):
    positive = 0
    negative = 0

    for g_sent, p_sent in zip(gold_sentences, pred_sentences):
        for g_token, p_token in zip(g_sent, p_sent):
            if g_token['dependency_head_id'] == p_token['dependency_head_id']:
                positive += 1
            else:
                negative += 1

    return positive/float(positive+negative)

print(unlabelled_attachment_score(gold, prediction))
