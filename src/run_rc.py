# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##set random seed
import numpy as np 
np.random.seed(26)
import tensorflow as tf
tf.set_random_seed(26)

import collections
import csv
import pandas as pd
import os,sys
import modeling
from tensorflow.contrib.layers.python.layers import initializers
import optimization
# import optimization_layerwise as optimization
# import accoptimization as optimization
import tokenization

import pickle
import codecs
from sklearn import metrics
from sklearn.externals import joblib

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("clean", True, "Whether to clean last training files.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, start_labels=None, end_labels=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.start_labels = start_labels
    self.end_labels = end_labels


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               start_labels_ids,
               end_labels_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_labels_ids = start_labels_ids
    self.end_labels_ids = end_labels_ids


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class NerProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train_raw.out")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_raw.out")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_raw.out")), "test")

  def get_labels(self):
    """See base class."""
    labels = ['0','1']
    return labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # if i == 0:
      #   continue
      # if set_type == 'train':
      #   if i > len(lines) * 0.01:
      #     continue
      # if set_type == 'dev': # filter unclear class
      #   if i > len(lines) * 0.01:
      #     continue
      # if set_type == 'test': # filter unclear class
      #   if i > len(lines) * 0.01:
      #     continue
      ## xy:先pad句子，再pad query
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1].strip())
      text_b = tokenization.convert_to_unicode(line[0].strip())
      # start_labels = line[2].strip()
      # end_labels = line[3].strip()
      start_labels = tokenization.convert_to_unicode(line[2].strip())
      end_labels = tokenization.convert_to_unicode(line[3].strip())
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, start_labels=start_labels, end_labels=end_labels))
    return examples

def convert_single_example(ex_index, example, label_map, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""


  all_start_labels = []
  all_end_labels = []

  text_a = example.text_a.split(' ')
  text_b = example.text_b
  start_labels = example.start_labels.split(' ')
  end_labels = example.end_labels.split(' ')
  

  # text_a_start_labels = []
  # text_a_end_labels = []

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  all_start_labels.append(0)
  all_end_labels.append(0)
  segment_ids.append(0)
  # print('**'*30)
  # print(len(text_a))
  # print(len(text_b))
  # print(len(start_labels))
  # print(len(end_labels))
  for i, word in enumerate(text_a):
    # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    tmp_s_label = start_labels[i]
    tmp_e_label = end_labels[i]
    ## 熊英：?? 对每个token处理是否存在问题？不应该是还要判断start和end吗
    ## xy revise
    for m in range(len(token)):
        if m == 0:
            all_start_labels.append(tmp_s_label)
            all_end_labels.append(0) # 不管怎样end都填0，如果是实体被token开，再最后判断把最后一个改成1
            segment_ids.append(0)  
        else:  # 一般不会出现else
            all_start_labels.append(0)
            all_end_labels.append(0) 
            segment_ids.append(0)
    if tmp_e_label == '1':
      # print('##################yyyyy################')
      all_end_labels[-1] = 1
  tokens.append("[SEP]")
  all_start_labels.append(0)
  all_end_labels.append(0)
  segment_ids.append(0)
 
  tokens_b = tokenizer.tokenize(text_b)
  
  for token in tokens_b:
    tokens.append(token)
    all_start_labels.append(0)
    all_end_labels.append(0)
    segment_ids.append(1)

  # all_start_labels.extend(text_a_start_labels)
  # all_end_labels.extend(text_a_end_labels)

  
          
  
  # 序列截断
  # 熊英：暴力截断会存在问题的吧？如果end_label为1，截断就没有对应的了？？
  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[:(max_seq_length - 1)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    all_start_labels = all_start_labels[:(max_seq_length - 1)]
    all_end_labels = all_end_labels[:(max_seq_length - 1)]
    segment_ids = segment_ids[:(max_seq_length - 1)]

  tokens.append("[SEP]")
  all_start_labels.append(0)
  all_end_labels.append(0)
  segment_ids.append(1)
          
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    # we don't concerned about it!
    all_start_labels.append(0)
    all_end_labels.append(0)
  
   
  all_start_labels_ids = [label_map[str(i)] for i in all_start_labels]
  all_end_labels_ids = [label_map[str(i)] for i in all_end_labels]


  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(all_start_labels_ids) == max_seq_length
  assert len(all_end_labels_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("start_labels_ids: %s" % " ".join([str(x) for x in all_start_labels_ids]))
    tf.logging.info("end_labels_ids: %s" % " ".join([str(x) for x in all_end_labels_ids]))


  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      start_labels_ids=all_start_labels_ids,
      end_labels_ids=all_end_labels_ids)
  return feature


def file_based_convert_examples_to_features(
    examples, label_map, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["start_labels_ids"] = create_int_feature(feature.start_labels_ids)
    features["end_labels_ids"] = create_int_feature(feature.end_labels_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "start_labels_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "end_labels_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:

      # d = d.repeat(1)
      d = d.shuffle(buffer_size=500)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn

def focal_loss(logits,labels,mask,num_labels,one_hot=True,lambda_param=1.5):
    probs = tf.nn.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    prob_label_pos = tf.where(tf.equal(labels,1),pos_probs,tf.ones_like(pos_probs))
    prob_label_neg = tf.where(tf.equal(labels,0),pos_probs,tf.zeros_like(pos_probs))
    loss = tf.pow(1. - prob_label_pos,lambda_param)*tf.log(prob_label_pos + 1e-7) + \
           tf.pow(prob_label_neg,lambda_param)*tf.log(1. - prob_label_neg + 1e-7)
    loss = -loss * tf.cast(mask,tf.float32)
    loss = tf.reduce_sum(loss,axis=-1,keepdims=True)
    # loss = loss/tf.cast(tf.reduce_sum(mask,axis=-1),tf.float32)
    loss = tf.reduce_mean(loss)
    return loss

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 start_labels_ids, end_labels_ids, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_sequence_output()
  hidden_size = output_layer.shape[-1].value
  max_seq_length = output_layer.shape[1].value

  ##add CRF layer and biLSTM layer
  if is_training:
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

  hidden = tf.reshape(output_layer, shape=[-1, hidden_size])
  with tf.variable_scope("start_logits"):
    start_W = tf.get_variable("start_W", shape=[hidden_size, num_labels],
                        dtype=tf.float32, initializer=initializers.xavier_initializer())

    start_b = tf.get_variable("start_b", shape=[num_labels], dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    
    start_pred = tf.nn.xw_plus_b(hidden, start_W, start_b)
  
  with tf.variable_scope("end_logits"):
    end_W = tf.get_variable("end_W", shape=[hidden_size, num_labels],
                        dtype=tf.float32, initializer=initializers.xavier_initializer())

    end_b = tf.get_variable("end_b", shape=[num_labels], dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    end_pred = tf.nn.xw_plus_b(hidden, end_W, end_b)
  
  with tf.variable_scope("start_loss"):
    logits = tf.reshape(start_pred, [-1, max_seq_length, num_labels])
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(start_labels_ids, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    start_loss = tf.reduce_mean(per_example_loss)
    probabilities = tf.nn.softmax(logits, axis=-1)
    start_pred_ids = tf.argmax(probabilities,axis=-1)
  
  with tf.variable_scope("end_start_loss"):
    logits = tf.reshape(end_pred, [-1, max_seq_length, num_labels])
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(end_labels_ids, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    end_loss = tf.reduce_mean(per_example_loss)
    probabilities = tf.nn.softmax(logits, axis=-1)
    end_pred_ids = tf.argmax(probabilities,axis=-1)

  total_loss = start_loss + end_loss
  return (total_loss, logits, start_pred_ids, end_pred_ids)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    start_labels_ids = features["start_labels_ids"]
    end_labels_ids = features["end_labels_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
    (total_loss, logits, start_pred_ids, end_pred_ids) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, start_labels_ids,
        end_labels_ids, num_labels, use_one_hot_embeddings)
    pred_ids = tf.stack([start_pred_ids,end_pred_ids],axis=1)
    print('-*'*30)
    print(pred_ids)
    
    tvars = tf.trainable_variables()
    scaffold_fn = None
    # 加载BERT模型
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                    init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    '''
    tf.logging.info("**** Trainable Variables ****")

    # 打印加载模型的参数
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    '''

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)  
    elif mode == tf.estimator.ModeKeys.EVAL:

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            scaffold_fn=scaffold_fn)  #
    else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=pred_ids,
            scaffold_fn=scaffold_fn
        )
    return output_spec

  return model_fn

def labeltoid(label_list):
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    # 保存label->index 的map
    with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map, w)

    return label_map

def save_best_model(cur_ckpt_path,best_model_path):
  cmd1 = 'cp '+cur_ckpt_path+'.index '+best_model_path+'.index'
  cmd2 = 'cp '+cur_ckpt_path+'.meta '+best_model_path+'.meta'
  cmd3 = 'cp '+cur_ckpt_path+'.data-00000-of-00001 '+best_model_path+'.data-00000-of-00001'
  os.system(cmd1)
  os.system(cmd2)
  os.system(cmd3)

def get_pred_metric(result, eval_input_ids, tokenizer):
  all_pred_ent = []
  # print(len(result))
  # print(len(eval_input_ids))
  # print(result)
  for i in range(len(result)):
    # print(i)
    tmp_input_ids = eval_input_ids[i]
    start_preds = result[i][0]
    end_preds = result[i][1]
    start_inds = []
    end_inds = []
    # print(start_preds)
    # print(end_preds)
    for ind in range(len(start_preds)):
      if(start_preds[ind]==1):
        start_inds.append(ind) 

    for ind in range(len(end_preds)):
      if(end_preds[ind]==1):
        end_inds.append(ind) 

    if(len(start_inds)==0):
      all_pred_ent.append('')
    else:
      ans = []
      def back(start_inds, end_inds):
          # global ans
          if(len(start_inds)==0 or len(end_inds)==0):
              return 
          while(len(end_inds)>0 and end_inds[0]<start_inds[0]):
              end_inds = end_inds[1:]     
          if(len(end_inds)>0):
              while(len(start_inds)>1 and (end_inds[0]-start_inds[1])>0 and ((end_inds[0]-start_inds[0])>(end_inds[0]-start_inds[1]))):
                  start_inds = start_inds[1:]
              ans.append((start_inds[0],end_inds[0]))
          back(start_inds[1:],end_inds[1:])
      back(start_inds, end_inds)
      if(len(ans)==0):
        all_pred_ent.append('')
      else:
        all_tmp_ent = []
        for item in ans:
          s_ind = item[0]
          e_ind = item[1]
          # print(s_ind, e_ind)
          tmp_ent = ' '.join(tokenizer.convert_ids_to_tokens(tmp_input_ids[s_ind:e_ind+1])).replace(' ##','')
          end_str = ''
          e_ind += 1
          while((e_ind<len(tmp_input_ids)-1) and ('##' in tokenizer.convert_ids_to_tokens([tmp_input_ids[e_ind]])[0])):
            end_str += tokenizer.convert_ids_to_tokens([tmp_input_ids[e_ind]])[0].replace('##','')
            e_ind += 1   
          tmp_ent += end_str
          all_tmp_ent.append(tmp_ent)
          # print(all_tmp_ent)
        all_pred_ent.append(all_tmp_ent)

        # print(' '.join(tokenizer.convert_ids_to_tokens(tmp_input_ids)))
        # print(all_tmp_ent)
  # print(all_pred_ent)
  # print(len(all_pred_ent))

  ## save result in file
  with open(os.path.join(FLAGS.output_dir, 'dev_pred_answer.txt'), 'w') as f:
    for entities in all_pred_ent:
      if len(entities) == 0:
        f.write('\n')
      else:
        f.write('\t'.join(entities) + '\n')

  with open(os.path.join(FLAGS.data_dir, 'dev_answer.txt'), 'r') as f:
    gold = f.readlines()
    


  all_pred = 0
  for item in all_pred_ent:
    if(item==''):
      continue 
    else:
      for i in item:
        all_pred += 1

  tp = 0
  all_ann = 0
  for i in range(len(gold)): 
    if(len(gold[i].strip())!=0):
      # print(gold[i])
      for k in gold[i].strip().split('\t'):
        all_ann += 1
  for i in range(len(gold)):
      if(all_pred_ent[i]!=''):
        for j in all_pred_ent[i]:
          for e in gold[i].strip().split('\t'):
            if j.lower() == e.lower():
              tp += 1
              break
  p = tp/all_pred
  r = tp/all_ann
  f = (2*p*r)/(p+r)
  f1 = f
  print(tp,all_pred,all_ann)
  print(p,r,f)
  # print(all_pred_ent)

  return f1



def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "ner": NerProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))
  ## del last training file  
  if(FLAGS.do_train and FLAGS.clean):     
      if os.path.exists(FLAGS.output_dir):
          def del_file(path):
              ls = os.listdir(path)
              for i in ls:
                  c_path = os.path.join(path, i)
                  if os.path.isdir(c_path):
                      del_file(c_path)
                  else:
                      os.remove(c_path)

          try:
              del_file(FLAGS.output_dir)
          except Exception as e:
              print(e)
              print('pleace remove the files of output dir and data.conf')
              exit(-1)


  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()
  label_map = labeltoid(label_list)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  print(tokenizer.convert_ids_to_tokens([101, 2424, 1996, 15316, 4668, 1997, 5423, 15660, 102 ]))
  # sys.exit(0)
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=None,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      model_dir=FLAGS.output_dir,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_map, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_input_ids = []
    for (ex_index, example) in enumerate(eval_examples):
      feature = convert_single_example(ex_index, example, label_map,
                                     FLAGS.max_seq_length, tokenizer)
      eval_input_ids.append(feature.input_ids)

    num_actual_eval_examples = len(eval_examples)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_map, FLAGS.max_seq_length, tokenizer, eval_file)
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

  ## Get id2label
  with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
      label2id = pickle.load(rf)
      id2label = {value: key for key, value in label2id.items()}

  best_result = 0
  all_results = []
  if FLAGS.do_train:
    for i in range(int(FLAGS.num_train_epochs)):
      print('**'*40)
      print('Train {} epoch'.format(i+1))
      estimator.train(input_fn=train_input_fn)
      ## Do Dev
    #   result = estimator.predict(input_fn=eval_input_fn)
    #   result = list(result)
    # for i in range(len(result)):
    #   tmp_input_ids = eval_input_ids[i]
    #   start_preds = result[i][0]
    #   end_preds = result[i][1]
    #   start_ind = 0
    #   end_ind = 0
    #   for ind in range(len(start_preds)):
    #     if(start_preds[ind]==1):
    #       start_ind = ind 
    #       break 
    #   for ind in range(len(end_preds)):
    #     if(end_preds[ind]==1):
    #       end_ind = ind 
    #       break 
    #   pred_ids = tmp_input_ids[start_ind:end_ind+1]
    #   print('-*-'*30)
    #   print(pred_ids)
    #   print(tokenizer.convert_ids_to_tokens(pred_ids))
      

      # pred = np.array(pred)
      # output_dev_file = os.path.join(FLAGS.output_dir, "label_dev.txt")
      # tmp_result = report_metric(eval_examples, pred, output_dev_file, id2label, probs)
      # print('Tmp result (macro F1) : ',tmp_result)
      # all_results.append(tmp_result)

      # if(tmp_result>best_result):
      #     print('**'*40)
      #     print('Found better model, saved!')
      #     best_result = tmp_result
      #     cur_ckpt_path = estimator.latest_checkpoint()
      #     best_model_path = '/'.join(cur_ckpt_path.split('/')[:-1]+['model.ckpt-best'])
      #     save_best_model(cur_ckpt_path,best_model_path)
    # print('**'*40)
    # print('Training completed!')
    # print('all_results: ',all_results)
    # print('Best result: ',np.max(all_results))
    # print('Avg result: ',np.mean(all_results))
  # import sys
  # sys.exit(0)    

  if FLAGS.do_predict:
    print('***********************Running Prediction************************')
    # print('Use model which perform best on dev data')
    cur_ckpt_path = estimator.latest_checkpoint()

    print('Use model which restore from last ckpt')
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        model_dir=None,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        warm_start_from=cur_ckpt_path)
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_map,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)
    result = list(result)

    print(get_pred_metric(result, eval_input_ids, tokenizer))

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
