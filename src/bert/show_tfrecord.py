import tensorflow as tf


# for example in tf.python_io.tf_record_iterator("./tf_examples.tfrecord"):
#     result = tf.train.Example.FromString(example)
#     print(result.features.feature['input_ids'].int64_list.value)
#     print(result.features.feature['input_mask'].int64_list.value)
#     print(result.features.feature['segment_ids'].int64_list.value)
#     print(result.features.feature['masked_lm_positions'].int64_list.value)
#     print(result.features.feature['masked_lm_ids'].int64_list.value)
#     print(result.features.feature['masked_lm_weights'].float_list.value)
#     print(result.features.feature['next_sentence_labels'].int64_list.value)


for idx, example in enumerate(tf.python_io.tf_record_iterator("./smiles.tfrecord")):
    result = tf.train.Example.FromString(example)


    # print(result.features.feature['input_ids'].int64_list.value)
    # print(result.features.feature['input_mask'].int64_list.value)
    # print(result.features.feature['masked_lm_positions'].int64_list.value)
    # print(result.features.feature['masked_lm_ids'].int64_list.value)
    # print(result.features.feature['masked_lm_weights'].float_list.value)
print(idx)
# 10k -> 6.4Mb
# 10m -> 6.4G
# 90m -> 57G

# 860k : 5.8G
# 10M : 67GB