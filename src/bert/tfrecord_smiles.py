import os
import csv
import json
import pickle
import random
import argparse
import numpy as np
import tensorflow as tf
import _pickle as cPickle
from copy import deepcopy
import collections
from collections import OrderedDict

__author__ = 'Bonggun Shin'


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_integer("max_seq_length", 100, "Maximum sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 15,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
# flags.DEFINE_string(
#     "output_file", "./smiles01.tfrecord,./smiles02.tfrecord",
#     "Output TF example file (or comma-separated list of files).")
flags.DEFINE_string("base_path", "../../../data/pretrain", "base path for dataset")
flags.DEFINE_string(
    "output_file", "%s/smiles.tfrecord" % FLAGS.base_path,
    "Output TF example file (or comma-separated list of files).")



def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(instance.tokens))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    # masked_lm_prob 0.15
    # max_seq_length", 170
    # max_predictions_per_seq", 26 (170*.15)
    # vocab_words = list(tokenizer.vocab.keys())
    # rng = random.Random(FLAGS.random_seed)
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[BEGIN]" or token == "[END]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(4, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


class SmilesTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        tokens = []
        tokens.append("[BEGIN]")
        for c in text:
            tokens.append(c)
        tokens.append("[END]")

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    def convert_by_vocab(self, vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = OrderedDict()
        index = 0
        with tf.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(self.tokens))
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(self.masked_lm_labels))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def truncate_seq_pair(tokens, max_num_tokens, rng):
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def read_smiles(base_path):
    smiles_list = []
    len_list = []
    with open("%s/pretrain/smiles_sample.csv" % (base_path), "rt") as f:
    # with open("%s/pretrain/00out_pubchem_cid_to_SMILES.csv" % (base_path), "rt") as f:  # 97,092,853 97M
        csvr = csv.reader(f, delimiter=',')
        for row in csvr:
            smiles_list.append(row[1])
            len_list.append(len(row[1]))

    tokenizer = SmilesTokenizer("%s/vocab_smiles.txt" % FLAGS.base_path)
    vocab_words = list(tokenizer.vocab.keys())
    rng = random.Random(12345)
    max_num_tokens = FLAGS.max_seq_length - 1

    instances = []
    for s in smiles_list:
        tokens = tokenizer.tokenize(s)
        truncate_seq_pair(tokens, max_num_tokens, rng)

        tokens.insert(0, "[CLS]")

        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens, FLAGS.masked_lm_prob,
                                                                                       FLAGS.max_predictions_per_seq,
                                                                                       vocab_words, rng)

        instance = TrainingInstance(
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        # print(instance)
        instances.append(instance)

    return instances, tokenizer


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='../../../data', help='Directory for input data.')
    args, unparsed = parser.parse_known_args()

    instances, tokenizer = read_smiles(args.base_path)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)

