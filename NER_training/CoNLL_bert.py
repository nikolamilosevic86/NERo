from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import codecs
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf
from tensorflow.contrib import rnn, cudnn_rnn

from tensorflow.contrib.tpu import TPUEstimator, TPUConfig
from tensorflow.python.ops import math_ops

import tf_metrics
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir. ex) 'NERdata'"
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. ex) 'bert_config.json'"
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train. ex) 'NER'"
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written. ex) 'output'"
)

flags.DEFINE_string(
    "init_checkpoint", "bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("use_crf", True, "Whether to use CRF decoding.")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_float("bert_dropout_rate", 0.2,
                   "Proportion of dropout for bert embedding.")

flags.DEFINE_float("bilstm_dropout_rate", 0.2,
                   "Proportion of dropout for bilstm.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("save_summary_steps", 100,
                     "Save summaries every this many steps")

flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "The maximum number of recent checkpoint files to keep")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on. ex) 'vocab.txt'")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string('data_config_path', 'data.conf',
                    'data config file, which save train and dev config')

flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, pos, chunk, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          pos: string. The pos(part of speech) tag of the example.
          chunk: string. The chunk tag of the example.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.pos = pos
        self.chunk = chunk
        self.label = label

        # untokenized data
        self.tokens = text.split()
        self.poss = pos.split()
        self.chunks = chunk.split()
        self.labels = []
        if label: self.labels = label.split()

        # tokenized data
        self.tokenized_tokens = []
        self.tokenized_poss = []
        self.tokenized_chunks = []
        self.tokenized_labels = []


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class NerProcessor(object):
    """A class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        '''
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "X", "[CLS]", "[SEP]"]
        '''
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "X"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):  # line = (w, p, c, l)
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            pos = tokenization.convert_to_unicode(line[1])
            chunk = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])
            examples.append(InputExample(guid=guid, text=text, pos=pos, chunk=chunk, label=label))
        return examples

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            poss = []
            chunks = []
            labels = []
            for line in f:
                contents = line.strip()
                if contents.startswith("-DOCSTART-"):
                    continue
                if len(contents) == 0:  # newline
                    if len(words) == 0: continue
                    assert (len(words) == len(poss))
                    assert (len(poss) == len(chunks))
                    assert (len(chunks) == len(labels))
                    w = ' '.join(words)
                    p = ' '.join(poss)
                    c = ' '.join(chunks)
                    l = ' '.join(labels)
                    lines.append([w, p, c, l])
                    words = []
                    poss = []
                    chunks = []
                    labels = []
                    continue
                tokens = line.strip().split(' ')
                assert (len(tokens) == 4)
                word = tokens[0]
                pos = tokens[1]
                chunk = tokens[2]
                label = tokens[-1]
                words.append(word)
                poss.append(pos)
                chunks.append(chunk)
                labels.append(label)
            return lines


def convert_single_example_to_feature(ex_index, example, label_map, max_seq_length, tokenizer, mode):
    textlist = example.tokens
    poslist = example.poss
    chunklist = example.chunks
    labellist = example.labels
    tokens = []
    poss = []
    chunks = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        pos_1 = poslist[i]
        chunk_1 = chunklist[i]
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                poss.append(pos_1)
                chunks.append(chunk_1)
                labels.append(label_1)
            else:
                poss.append("X")
                chunks.append("X")
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        poss = poss[0:(max_seq_length - 2)]
        chunks = chunks[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    # save tokens, poss, chunks, labels back to example
    example.tokenized_tokens = tokens
    example.tokenized_poss = poss
    example.tokenized_chunks = chunks
    example.tokenized_labels = labels

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # label_ids.append(label_map["[CLS]"])
    label_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # label_ids.append(label_map["[SEP]"])
    label_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    return feature


def convert_feature_to_tf_example(feature):
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    # build labe2id.pkl
    label_map = {}
    for (i, label) in enumerate(label_list, 1):  # 0 index for '0' padding
        label_map[label] = i
    with codecs.open('./output/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)

    # convert examples => features => tf_examples(tf.train.Example) => tf records(TFRecord, file)
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example_to_feature(ex_index, example, label_map, max_seq_length, tokenizer, mode)
        tf_example = convert_feature_to_tf_example(feature)
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    embedding = model.get_sequence_output()  # (batch_size, seq_length, embedding_size)
    '''
    embedding_1 = model.get_all_encoder_layers()[-2]
    embedding_2 = model.get_all_encoder_layers()[-1]
    embedding = tf.concat([embedding_1, embedding_2], axis=-1)
    '''
    if is_training:
        # dropout embedding
        embedding = tf.layers.dropout(embedding, rate=FLAGS.bert_dropout_rate, training=is_training)
    embedding_size = embedding.shape[-1].value  # embedding_size
    seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # (batch_size)
    print('seq_length', seq_length)
    print('lengths', lengths)

    def bi_lstm_fused(inputs, lengths, rnn_size, is_training, dropout_rate=0.5, scope='bi-lstm-fused'):
        with tf.variable_scope(scope):
            t = tf.transpose(inputs, perm=[1, 0, 2])  # Need time-major
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
            return tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

    def lstm_layer(inputs, lengths, is_training):
        rnn_output = tf.identity(inputs)
        for i in range(2):
            scope = 'bi-lstm-fused-%s' % i
            rnn_output = bi_lstm_fused(rnn_output,
                                       lengths,
                                       rnn_size=FLAGS.lstm_size,
                                       is_training=is_training,
                                       dropout_rate=FLAGS.bilstm_dropout_rate,
                                       scope=scope)  # (batch_size, seq_length, 2*rnn_size)
        return rnn_output

    def project_layer(inputs, out_dim, seq_length, scope='project'):
        with tf.variable_scope(scope):
            in_dim = inputs.get_shape().as_list()[-1]
            weight = tf.get_variable('W', shape=[in_dim, out_dim],
                                     dtype=tf.float32, initializer=initializers.xavier_initializer())
            bias = tf.get_variable('b', shape=[out_dim], dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
            t_output = tf.reshape(inputs, [-1, in_dim])  # (batch_size*seq_length, in_dim)
            output = tf.matmul(t_output, weight) + bias  # (batch_size*seq_length, out_dim)
            output = tf.reshape(output, [-1, seq_length, out_dim])  # (batch_size, seq_length, out_dim)
            return output

    def loss_layer(logits, labels, num_labels, lengths, input_mask):
        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=initializers.xavier_initializer())
        if FLAGS.use_crf:
            with tf.variable_scope("crf-loss"):
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=labels,
                    transition_params=trans,
                    sequence_lengths=lengths)
                per_example_loss = -log_likelihood
                loss = tf.reduce_mean(per_example_loss)
                return loss, per_example_loss, trans
        else:
            labels_one_hot = tf.one_hot(labels, num_labels)
            cross_entropy = labels_one_hot * tf.log(tf.nn.softmax(logits))
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            cross_entropy *= tf.to_float(input_mask)
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(lengths, tf.float32)
            per_example_loss = cross_entropy
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, trans

    '''    
    # 1
    logits = project_layer(embedding, num_labels, seq_length, scope='project')
    '''
    '''
    # 2
    lstm_outputs = lstm_layer(embedding, lengths, is_training)
    p1 = project_layer(lstm_outputs, FLAGS.lstm_size, seq_length, scope='project-1')
    p2 = project_layer(p1, num_labels, seq_length, scope='project-2')
    logits = p2
    '''
    # 3
    lstm_outputs = lstm_layer(embedding, lengths, is_training)
    logits = project_layer(lstm_outputs, num_labels, seq_length, scope='project')
    loss, per_example_loss, trans = loss_layer(logits, labels, num_labels, lengths, input_mask)
    if FLAGS.use_crf:
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)
    else:
        probabilities = tf.nn.softmax(logits, axis=-1)
        pred_ids = tf.argmax(probabilities, axis=-1)

    # masking for confirmation
    pred_ids *= input_mask

    print('#' * 20)
    print('shape of output_layer:', embedding.shape)
    print('embedding_size:%d' % embedding_size)
    print('seq_length:%d' % seq_length)
    print('shape of logit', logits.shape)
    print('shape of loss', loss.shape)
    print('shape of per_example_loss', per_example_loss.shape)
    print('num labels:%d' % num_labels)
    print('#' * 20)
    return (loss, per_example_loss, logits, trans, pred_ids)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        print('shape of label_ids', label_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        print('shape of pred_ids', pred_ids.shape)

        global_step = tf.train.get_or_create_global_step()
        # add summary
        tf.summary.scalar('loss', total_loss)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint and is_training:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids,
                scaffold_fn=scaffold_fn
            )
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                '''
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                '''
                lr = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.9, staircase=True)
                optimizer = tf.train.AdamOptimizer(lr)
                grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 1.5)
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
                logging_hook = tf.train.LoggingTensorHook({"batch_loss": total_loss}, every_n_iter=10)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[logging_hook],
                    scaffold_fn=scaffold_fn)
            else:  # mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(label_ids, pred_ids, per_example_loss, input_mask):
                    # ['<pad>'] + ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "X"]
                    indices = [2, 3, 4, 5, 6, 7, 8, 9]
                    precision = tf_metrics.precision(label_ids, pred_ids, num_labels, indices, input_mask)
                    recall = tf_metrics.recall(label_ids, pred_ids, num_labels, indices, input_mask)
                    f = tf_metrics.f1(label_ids, pred_ids, num_labels, indices, input_mask)
                    accuracy = tf.metrics.accuracy(label_ids, pred_ids, input_mask)
                    loss = tf.metrics.mean(per_example_loss)
                    return {
                        'eval_precision': precision,
                        'eval_recall': recall,
                        'eval_f': f,
                        'eval_accuracy': accuracy,
                        'eval_loss': loss,
                    }

                eval_metrics = (metric_fn, [label_ids, pred_ids, per_example_loss, input_mask])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_summary_steps=FLAGS.save_summary_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}

    if FLAGS.do_train:
        if len(data_config) == 0:
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            num_train_steps = int((len(train_examples) / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
            data_config['num_train_steps'] = num_train_steps
            data_config['num_warmup_steps'] = num_warmup_steps
            data_config['num_train_size'] = len(train_examples)
        else:
            num_train_steps = int(data_config['num_train_steps'])
            num_warmup_steps = int(data_config['num_warmup_steps'])

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,  # 1 for '0' padding
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # prepare train_input_fn
        if data_config.get('train.tf_record_path', '') == '':
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            filed_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        else:
            train_file = data_config.get('train.tf_record_path')
        num_train_size = num_train_size = int(data_config['num_train_size'])
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", num_train_size)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # prepare eval_input_fn
        if data_config.get('eval.tf_record_path', '') == '':
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
            data_config['eval.tf_record_path'] = eval_file
            data_config['num_eval_size'] = len(eval_examples)
        else:
            eval_file = data_config['eval.tf_record_path']
        num_eval_size = data_config.get('num_eval_size', 0)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", num_eval_size)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(num_eval_size / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        # train and evaluate
        hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator, 'eval_f', 3000, min_steps=30000, run_every_secs=120)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120)
        tp = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        result = tp[0]

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if not os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path, 'a', encoding='utf-8') as fd:
            json.dump(data_config, fd)

    if FLAGS.do_predict:
        # prepare predict_input_fn
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        # predict
        predict_steps = None
        if FLAGS.use_tpu:
            predict_steps = int(len(predict_examples) / FLAGS.eval_batch_size)
        predicted_result = estimator.evaluate(input_fn=predict_input_fn, steps=predict_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "predicted_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Predict results *****")
            for key in sorted(predicted_result.keys()):
                tf.logging.info("  %s = %s", key, str(predicted_result[key]))
                writer.write("%s = %s\n" % (key, str(predicted_result[key])))

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "pred.txt")

        print('*' * 20)
        print('type of result:%s, type of predict_examples:%s' % (type(result), type(predict_examples)))
        print('*' * 20)
        with codecs.open('./output/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            for predict_example, prediction in zip(predict_examples, result):
                tokens = predict_example.tokens
                poss = predict_example.poss
                chunks = predict_example.chunks
                labels = predict_example.labels
                tokenized_tokens = predict_example.tokenized_tokens
                tokenized_poss = predict_example.tokenized_poss
                tokenized_chunks = predict_example.tokenized_chunks
                tokenized_labels = predict_example.tokenized_labels
                text = predict_example.text
                length = len(tokenized_tokens)
                seq = 0
                for token, pos, label, p_id in zip(tokenized_tokens, tokenized_poss, tokenized_labels,
                                                   prediction[1:length + 1]):
                    p_label = 'O'
                    if p_id != 0: p_label = id2label[p_id]
                    if p_label == 'X': p_label = 'O'
                    if label == 'X': continue
                    org_token = tokens[seq]
                    org_pos = poss[seq]
                    org_chunk = chunks[seq]
                    org_label = labels[seq]
                    output_line = ' '.join([org_token, org_pos, org_chunk, org_label, p_label])
                    writer.write(output_line + '\n')
                    seq += 1
                writer.write('\n')


def data_load():
    processer = NerProcessor()
    processer.get_labels()
    processer.get_train_examples(FLAGS.data_dir)
    print()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
tf.app.run()