import argparse
import numpy as np
import _pickle as cPickle
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate
from keras.layers import Dense, Dropout, Input, Embedding, Flatten
from keras.models import Model
from keras.layers import LSTM, Bidirectional
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


__author__ = 'Bonggun Shin'


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def get_trn_dev(dataset, smiles_bert_cls, fold=0):
    (XD, XT, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx) = dataset
    trn_set = trn_sets[fold]
    dev_set = dev_sets[fold]

    drug_idx_trn = row_idx[trn_set]
    protein_idx_trn = col_idx[trn_set]
    drug_idx_dev = row_idx[dev_set]
    protein_idx_dev = col_idx[dev_set]

    (xd_trn, xt_trn, y_trn) = prepare_interaction_pairs(smiles_bert_cls, XT, Y, drug_idx_trn, protein_idx_trn)
    (xd_dev, xt_dev, y_dev) = prepare_interaction_pairs(smiles_bert_cls, XT, Y, drug_idx_dev, protein_idx_dev)
    trndev = xd_trn, xt_trn, y_trn, xd_dev, xt_dev, y_dev

    return trndev


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select


def get_model():
    max_len_d = 40
    max_len_t = 1000
    n_vocab_d = 64
    n_vocab_t = 25
    n_filter = 32
    d_filter_size = 8
    t_filter_size = 12

    xd_input = Input(shape=(100,40), dtype='float32')
    xt_input = Input(shape=(max_len_t,), dtype='int32')

    # xd_z = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(xd_input)
    xd_z = Flatten()(xd_input)

    xt_emb = Embedding(input_dim=n_vocab_t + 1, output_dim=128)(xt_input)
    xt_z = Conv1D(filters=n_filter, kernel_size=t_filter_size, activation='relu', padding='valid', strides=1)(xt_emb)
    xt_z = Conv1D(filters=n_filter * 2, kernel_size=t_filter_size, activation='relu', padding='valid', strides=1)(xt_z)
    xt_z = Conv1D(filters=n_filter * 3, kernel_size=t_filter_size, activation='relu', padding='valid', strides=1)(xt_z)
    xt_z = GlobalMaxPooling1D()(xt_z)

    concat_z = concatenate([xd_z, xt_z])

    z = Dense(1024, activation='relu')(concat_z)
    z = Dropout(0.1)(z)
    z = Dense(1024, activation='relu')(z)
    z = Dropout(0.1)(z)
    z = Dense(512, activation='relu')(z)

    output = Dense(1, kernel_initializer='normal')(z)
    model = Model(inputs=[xd_input, xt_input], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])
    model.summary()

    return model


def get_session(gpu_fraction=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='../../../data', help='Directory for input data.')
    args, unparsed = parser.parse_known_args()
    ktf.set_session(get_session())

    batch_size = 256
    epochs = 2000

    dataset = cPickle.load(open('%s/kiba/kiba.cpkl' % args.base_path, 'rb'))
    # smiles_bert_cls = cPickle.load(open('./kiba_smiles_bert.cpkl', 'rb'))
    smiles_bert_cls = cPickle.load(open('./kiba_smiles_bert_full.cpkl', 'rb'))


    trndev = get_trn_dev(dataset, smiles_bert_cls, 0)
    xd_trn, xt_trn, y_trn, xd_dev, xt_dev, y_dev = trndev
    model = get_model()
    model.fit([xd_trn, xt_trn], y_trn,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[],
              epochs=epochs,
              validation_data=([xd_dev, xt_dev], y_dev))



