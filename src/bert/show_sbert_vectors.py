import json
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import _pickle as cPickle

input_file = "./output.jsonl"
# vectors = json.load(open(input_file), object_pairs_hook=OrderedDict)
#
# vector_list = []
# with tf.gfile.Open(input_file, "r") as lines:
#     for line in lines:
#         parsed = json.loads(line)
#         uid = parsed['linex_index']
#         features = parsed['features']
#
#         smiles = ''.join([f['token'] for f in features])
#
#         f_cls = features[0]
#         l1 = f_cls['layers'][0]['values']
#         l2 = f_cls['layers'][1]['values']
#
#         vector = np.concatenate((l1, l2))
#         vector_list.append(vector)
#
# cPickle.dump(np.array(vector_list), open('./kiba_smiles_bert.cpkl', 'wb'))
#
# print(1)


vector_list = []
with tf.gfile.Open(input_file, "r") as lines:
    for line in lines:
        parsed = json.loads(line)
        uid = parsed['linex_index']
        features = parsed['features']

        smiles = ''.join([f['token'] for f in features])

        tokens = []
        for i, f in enumerate(features):
            l1 = f['layers'][0]['values']
            l2 = f['layers'][1]['values']
            vector = np.concatenate((l1, l2))
            tokens.append(vector)

        for k in range(i, 99):
            tokens.append(np.zeros((40,)))

        vector_list.append(tokens)

cPickle.dump(np.array(vector_list), open('./kiba_smiles_bert_full.cpkl', 'wb'))

print(1)

