# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import functools
import os
import tempfile

# pylint: disable=g-bad-import-order
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import flags
import numpy as np
# pylint: enable=g-bad-import-order

from tensorflow.python.keras.distribute import distributed_training_utils
from official.transformer import compute_bleu
from official.transformer.utils import schedule
from official.transformer.utils import tokenizer
from official.transformer.v2 import dataset
from official.transformer.v2 import metrics
from official.transformer.v2 import misc
from official.transformer.v2 import optimizer
from official.transformer.v2 import tf_importer
from official.transformer.v2 import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
tf = tf_importer.tf


INF = int(1e9)
BLEU_DIR = "bleu"


def _validate_file(filepath):
  """Make sure that file exists."""
  if not tf.io.gfile.exists(filepath):
    raise tf.errors.NotFoundError(None, None, "File %s not found." % filepath)


def translate_and_compute_bleu(model, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def evaluate_and_log_bleu(model, bleu_source, bleu_ref, vocab_file):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, subtokenizer, bleu_source, bleu_ref)

  tf.compat.v1.logging.info("Bleu score (uncased): %s", uncased_score)
  tf.compat.v1.logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score


def all_local_devices(num_gpus):
  return (tuple("/device:GPU:%d" % i for i in range(num_gpus)) or
          ("/device:CPU:0",))


class TransformerMain(object):

  def __init__(self, flags_obj):
    """Args:

      flags_obj: Object containing parsed flag values.
    """
    if flags_obj.no_random:
      np.random.seed(1)
      tf_importer._tf.set_random_seed(1)
    if flags_obj.model_version == "eager":
      tf.compat.v1.enable_eager_execution()
    self.flags_obj = flags_obj

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["tpu"] = flags_obj.tpu
    params["use_tpu"] = bool(flags_obj.tpu)  # was a tpu specified.
    params["static_batch"] = flags_obj.static_batch
    params["allow_ffn_pad"] = not params["use_tpu"]

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["no_random"] = flags_obj.no_random

    # Set batch size parameter, which depends on the availability of
    # TPU and GPU, and distribution settings.
    params["batch_size"] = (
        flags_obj.batch_size or params["default_batch_size_tpu"])

    params["repeat_dataset"] = True
    """self.schedule_manager = schedule_manager = schedule.Manager(

        train_steps=flags_obj.train_steps,
        steps_between_evals=flags_obj.steps_between_evals,
        train_epochs=flags_obj.train_epochs,
        epochs_between_evals=flags_obj.epochs_between_evals,
        default_train_epochs=flags_obj.train_epochs,
        batch_size=params["batch_size"],
        max_length=params["max_length"],
        use_tpu=params["use_tpu"],
        num_tpu_shards=flags_obj.num_tpu_shards)
    """

    # Train and evaluate transformer model
    version = flags_obj.model_version
    if flags_obj.mode == "train":
      # tf.keras.backend.set_learning_phase(1)
      self.train(flags_obj, version)
    elif flags_obj.mode == "predict":
      self.predict(flags_obj, version)
    elif flags_obj.mode == "eval":
      self.eval(flags_obj, version)
    elif flags_obj.mode == "one_step":
      self.train_one_step(flags_obj, version)
    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))

  def train_one_step(self, flags_obj, version):
    params, is_train = self.params, True
    assert(version == "keras")
    model_dict = transformer.create_model(params, is_train)
    model = model_dict["model"]
    targets, logits = model_dict["targets"], model_dict["logits"]
    get_pred_fn = lambda y_label, y_pred: y_pred
    opt = self._create_optimizer_v2()
    model.compile(
        opt, loss={"transformer_loss": get_pred_fn}, target_tensors=[])
        # Add this parameter to enable Mirrored DS on subclassed models
    self._load_weights_if_possible(model, flags_obj.init_weight_path)
    model.summary()
    map_data_fn = lambda x, y: ((x, y), tf.constant(0.0))
    ds = dataset.train_input_fn(params)
    ds = ds.map(map_data_fn, num_parallel_calls=params["num_parallel_calls"])
    x, y = tf.data.experimental.get_single_element(ds.take(1))
    x1, x2 = x
    try:
      sess = tf.compat.v1.Session()
      with sess.as_default():
        x1n = x1.eval()
        x2n = x2.eval()
        print(x1n, x2n, y.eval())
    except NotImplementedError:  # eval is not supported in eager
      x1n = x1.numpy()
      x2n = x2.numpy()
      print(x1n, x2n, y.numpy())
    init_epoch = 0 if flags_obj.init_epoch is None else flags_obj.init_epoch
    init_steps = init_epoch * flags_obj.steps_between_evals

    sfunc = functools.partial(
        optimizer.get_learning_rate,
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(
        sfunc, init_steps, verbose=False)

    callbacks = [
        scheduler_callback,
    ]

    valid_ds = dataset.eval_input_fn(params)
    valid_ds = valid_ds.map(
        map_data_fn, num_parallel_calls=params["num_parallel_calls"])

    history = model.fit(
        ds,
        initial_epoch=init_epoch,
        epochs=1,
        steps_per_epoch=1,
        callbacks=callbacks)
    logits = model.predict(x=(x1n, x2n))
    try:
      with sess.as_default():
        print(logits.eval())
    except NotImplementedError:  # eval is not supported in egaer
      print(logits.numpy())


  def train(self, flags_obj, version):
    params, is_train = self.params, True

    if version == "eager":
      tf.compat.v1.logging.info("=== Build Eager mode.")
      model_dict = transformer.create_model(params, is_train)
      model = model_dict["model"]
      ds = dataset.train_input_fn(params)
      self.run_loop()
    elif version == "keras":
      tf.compat.v1.logging.info("=== Build Keras mode.")
      model_dict = transformer.create_model(params, is_train)
      model = model_dict["model"]
      targets, logits = model_dict["targets"], model_dict["logits"]
      '''
      metric_dict = metrics.create_v2_metrics(self.params["vocab_size"])
      for k, metric_fn in metric_dict.items():
        model.add_metric(metric_fn(targets, logits), name=k)
      '''
      get_pred_fn = lambda y_label, y_pred: y_pred
      opt = self._create_optimizer_v2()
      model.compile(
          opt, loss={"transformer_loss": get_pred_fn}, target_tensors=[])
          # Add this parameter to enable Mirrored DS on subclassed models
      self._load_weights_if_possible(model, flags_obj.init_weight_path)
      model.summary()

      map_data_fn = lambda x, y: ((x, y), tf.constant(0.0))
      ds = dataset.train_input_fn(params)
      ds = ds.map(map_data_fn, num_parallel_calls=params["num_parallel_calls"])
      init_epoch = 0 if flags_obj.init_epoch is None else flags_obj.init_epoch
      init_steps = init_epoch * flags_obj.steps_between_evals
      trains_epochs = DEFAULT_TRAIN_EPOCHS if flags_obj.train_epochs is None else flags_obj.train_epochs

      sfunc = functools.partial(
          optimizer.get_learning_rate,
          learning_rate=params["learning_rate"],
          hidden_size=params["hidden_size"],
          learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
      scheduler_callback = optimizer.LearningRateScheduler(
          sfunc, init_steps, verbose=False)

      if flags_obj.init_logdir_timestamp is not None:
        timestamp = flags_obj.init_logdir_timestamp
      else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
      cur_log_dir = os.path.join(flags_obj.model_dir, timestamp)
      if not os.path.exists(cur_log_dir):
        os.makedirs(cur_log_dir)
      tb_logdir = os.path.join(cur_log_dir, "logs")
      tb_callbacks = tf.keras.callbacks.TensorBoard(tb_logdir)
      save_path = os.path.join(
          cur_log_dir, "weights-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5")
      csv_path = os.path.join(cur_log_dir, "result.csv")
      callbacks = [
          scheduler_callback,
          tb_callbacks,
          tf.keras.callbacks.ModelCheckpoint(save_path, save_weights_only=True),
          tf.keras.callbacks.CSVLogger(csv_path, append=True),
      ]

      valid_ds = dataset.eval_input_fn(params)
      valid_ds = valid_ds.map(
          map_data_fn, num_parallel_calls=params["num_parallel_calls"])

      history = model.fit(
          ds,
          initial_epoch=init_epoch,
          epochs=trains_epochs,
          steps_per_epoch=flags_obj.steps_between_evals,
          validation_data=valid_ds,
          validation_steps=64,
          callbacks=callbacks)
      tf.compat.v1.logging.info("\nTrain history: {}".format(history.history))

      save_weight_path = os.path.join(cur_log_dir, "saves-model-weights.hdf5")
      save_model_path = os.path.join(cur_log_dir, "saves-model.hdf5")
      model.save_weights(save_weight_path)
      model.save(save_model_path)

  def eval(self, flags_obj, version):
    params, is_train = self.params, False

    tf.compat.v1.logging.info("=== Build Keras eval.")
    with tf.name_scope("model"):
      model_dict = transformer.create_model(params, is_train)
      model = model_dict["model"]
      self._load_weights_if_possible(model, flags_obj.init_weight_path)
      model.summary()
    tf.compat.v1.logging.info("=== Run Keras eval.")
    evaluate_and_log_bleu(model, flags_obj.bleu_source, flags_obj.bleu_ref,
                          flags_obj.vocab_file)

  def predict(self, flags_obj, version):
    params, is_train = self.params, False

    tf.compat.v1.logging.info("Keras eval")
    with tf.name_scope("model"):
      model_dict = transformer.create_model(params, is_train)
      model = model_dict["model"]
      self._load_weights_if_possible(model, flags_obj.init_weight_path)
      model.summary()
    subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)
    SINGLE_BATCH = 1
    ds = dataset.eval_input_fn(params)
    ds = ds.map(lambda x, y: x).take(SINGLE_BATCH)
    ret = model.predict(ds)
    val_outputs, val_scores = ret
    tf.compat.v1.logging.info("Predicts result: {}".format(ret))
    tf.compat.v1.logging.info("outputs: {}".format(val_outputs.shape))
    tf.compat.v1.logging.info("scores: {}".format(val_scores.shape))
    length = len(val_outputs)
    for i in range(length):
      translate.translate_from_input(val_outputs[i], subtokenizer)

  def _load_weights_if_possible(self, model, init_weight_path=None):
    if init_weight_path:
      tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
      model.load_weights(init_weight_path, by_name=True)

  def _create_distribution_strategy(self):
    strat = distribution_utils.get_distribution_strategy(
        distribution_strategy="mirrored", num_gpus=8)
    print('========================================= ', strat)
    return strat

  def _create_optimizer(self, global_step):
    params = self.params
    learning_rate = get_learning_rate(
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        learning_rate_warmup_steps=params["learning_rate_warmup_steps"],
        global_step=global_step)

    opt = tf.contrib.opt.LazyAdamOptimizer(
        0.1,
        beta1=params["optimizer_adam_beta1"],
        beta2=params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])
    return opt

  def _create_optimizer_v2(self):
    params = self.params
    opt = optimizer.LazyAdam(
        0.1,
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])
    return opt

  def run_loop(self):
    model = self.model
    # Training arguments

    # BLEU calculation arguments
    bleu_source = self.flags_obj.bleu_source
    bleu_ref = self.flags_obj.bleu_ref
    bleu_threshold = self.flags_obj.stop_threshold
    vocab_file = self.flags_obj.vocab_file
    model_dir = self.params["model_dir"]

    if bleu_source:
      _validate_file(bleu_source)
    if bleu_ref:
      _validate_file(bleu_ref)
    if vocab_file:
      _validate_file(vocab_file)

    evaluate_bleu = bleu_source is not None and bleu_ref is not None

    # Print details of training schedule.
    tf.compat.v1.logging.info("Training schedule:")
    tf.compat.v1.logging.info("\t1. Train for {}".format(
        schedule_manager.train_increment_str))
    tf.compat.v1.logging.info("\t2. Evaluate model.")
    if evaluate_bleu:
      tf.compat.v1.logging.info("\t3. Compute BLEU score.")
      if bleu_threshold is not None:
        tf.compat.v1.logging.info(
            "Repeat above steps until the BLEU score reaches %f" %
            bleu_threshold)
    if not evaluate_bleu or bleu_threshold is None:
      tf.compat.v1.logging.info("Repeat above steps %d times." %
                                schedule_manager.train_eval_iterations)

    if evaluate_bleu:
      # Create summary writer to log bleu score (values can be displayed in
      # Tensorboard).
      # bleu_writer = tf.summary.FileWriter(os.path.join(model_dir, BLEU_DIR))
      if bleu_threshold is not None:
        # Change loop stopping condition if bleu_threshold is defined.
        schedule_manager.train_eval_iterations = INF

    tf.compat.v1.logging.info("steps_between_evals: %s",
                              self.flags_obj.steps_between_evals)
    tf.compat.v1.logging.info("train_steps: %s", self.params["train_steps"])
    tf.compat.v1.logging.info("repeat_dataset: %s",
                              self.params["repeat_dataset"])

    use_gpu = tf.test.is_gpu_available()
    global_step = tf.zeros([], dtype=tf.int64, name="global_step")
    opt = self._create_optimizer(global_step)
    # Loop training/evaluation/bleu cycles
    for i in xrange(schedule_manager.train_eval_iterations):
      tf.compat.v1.logging.info("Starting iteration %d" % (i + 1))

      ds = dataset.train_input_fn(self.params)
      for j, (features, labels) in enumerate(ds):
        tf.compat.v1.logging.info("internal train iteration: %s", j)
        tf.compat.v1.logging.info("use_gpu: %s", use_gpu)
        tf.compat.v1.logging.info("gpu_device_name: %s",
                                  tf.test.gpu_device_name())

        with tf.GradientTape() as tape:
          # features = tf.keras.layers.Input((None))
          # labels = tf.keras.layers.Input((None))
          if use_gpu:
            features = features.gpu()
            tf.compat.v1.logging.info("use gpu")
          tf.compat.v1.logging.info("feature %s", features.shape)
          tf.compat.v1.logging.info("labels %s", labels.shape)
          logits = self.model(features, targets=labels)
          # tf.compat.v1.logging.info('logits %s', logits)
          loss = metrics.transformer_loss(
              logits,
              labels,
              smoothing=self.params["label_smoothing"],
              vocab_size=self.params["vocab_size"])
        tvars = self.model.trainable_weights
        grad = tape.gradient(loss, tvars)
        opt.apply_gradients(zip(grad, tvars), name="train")
        global_step += 1

        if j >= self.flags_obj.steps_between_evals:
          break

      # Train the model for single_iteration_train_steps or until the input fn
      # runs out of examples (if single_iteration_train_steps is None).
      ds = dataset.eval_input_fn(self.params)
      m = metrics.EvalMetricsV2(self.params)
      for j, (features, labels) in enumerate(ds):
        logits = self.model(features, targets=labels)
        tf.compat.v1.logging.info("internal eval iteration: %s", j)
        # tf.compat.v1.logging.info('logits: %s', logits)
        m(logits, labels)
        tf.compat.v1.logging.info("metric: %s", m.result())
        if j >= 1:
          break


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    TransformerMain(flags.FLAGS)


if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  misc.define_transformer_flags()
  tf.compat.v1.app.run(main)

