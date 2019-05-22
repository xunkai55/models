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

import datetime
import os
import tempfile

# TODO(tianlin) Import internal library. Remove this when different behaviors
# of keras_model.fit(dataset, ...) for different TF versions are fixed.
from tensorflow.python import tf2 as tf2_internal

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer import compute_bleu
from official.transformer.utils import tokenizer
from official.transformer.v2 import data_pipeline
from official.transformer.v2 import misc
from official.transformer.v2 import optimizer
from official.transformer.v2 import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils


INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1


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


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.
    """
    self.flags_obj = flags_obj

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)
    params["num_gpus"] = num_gpus
    params["no_dist_strat"] = flags_obj.no_dist_strat
    params["multi_worker_strat"] = flags_obj.multi_worker_strat

    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["static_batch"] = flags_obj.static_batch
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None

  def train(self):
    """Trains the model."""
    with self._create_distribution_strategy().scope():
      params, flags_obj, is_train = self.params, self.flags_obj, True
      model = transformer.create_model(params, is_train)
      opt = self._create_optimizer()

      model.compile(opt, target_tensors=[])
      model.summary()
      self._load_weights_if_possible(model, flags_obj.init_weight_path)

      cur_log_dir = _get_log_dir_or_default(flags_obj)
      _ensure_dir(cur_log_dir)

      if tf2_internal.enabled():
        map_data_fn = data_pipeline.map_data_for_transformer_fn_tf2
      else:
        map_data_fn = data_pipeline.map_data_for_transformer_fn_tf1
      train_ds = data_pipeline.train_input_fn(params)
      train_ds = train_ds.map(
          map_data_fn, num_parallel_calls=params["num_parallel_calls"])
      valid_ds = data_pipeline.eval_input_fn(params)
      valid_ds = valid_ds.map(
          map_data_fn, num_parallel_calls=params["num_parallel_calls"])

      init_epoch = flags_obj.init_epoch or 0
      init_steps = init_epoch * flags_obj.steps_per_epoch
      callbacks = self._create_callbacks(cur_log_dir, init_steps, params)

      history = model.fit(
          train_ds,
          initial_epoch=init_epoch,
          epochs=flags_obj.train_epochs,
          steps_per_epoch=flags_obj.steps_per_epoch,
          validation_data=valid_ds,
          validation_steps=flags_obj.validation_steps,
          callbacks=callbacks)
      tf.compat.v1.logging.info("\nTrain history: {}".format(history.history))

      save_weight_path = os.path.join(cur_log_dir, "saves-model-weights.hdf5")
      save_model_path = os.path.join(cur_log_dir, "saves-model.hdf5")
      model.save_weights(save_weight_path)
      model.save(save_model_path)

  def eval(self):
    """Evaluates the model."""
    params, flags_obj, is_train = self.params, self.flags_obj, False
    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train)
      self._load_weights_if_possible(model, flags_obj.init_weight_path)
      model.summary()
    evaluate_and_log_bleu(model, flags_obj.bleu_source, flags_obj.bleu_ref,
                          flags_obj.vocab_file)

  def predict(self):
    """Predicts result from the model."""
    params, flags_obj, is_train = self.params, self.flags_obj, False

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train)
      self._load_weights_if_possible(model, flags_obj.init_weight_path)
      model.summary()
    subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)

    ds = data_pipeline.eval_input_fn(params)
    ds = ds.map(lambda x, y: x).take(_SINGLE_SAMPLE)
    ret = model.predict(ds)
    val_outputs, _ = ret
    length = len(val_outputs)
    for i in range(length):
      translate.translate_from_input(val_outputs[i], subtokenizer)

  def _create_callbacks(self, cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"] * params["num_gpus"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)

    tb_logdir = os.path.join(cur_log_dir, "logs")
    save_path = os.path.join(cur_log_dir,
                             "weights-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5")
    csv_path = os.path.join(cur_log_dir, "result.csv")
    return [
        scheduler_callback,
        tf.keras.callbacks.TensorBoard(tb_logdir),
        tf.keras.callbacks.ModelCheckpoint(save_path, save_weights_only=True),
        tf.keras.callbacks.CSVLogger(csv_path, append=True),
    ]

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
      model.load_weights(init_weight_path, by_name=True)

  def _create_distribution_strategy(self):
    if self.params["no_dist_strat"]:
      return misc.DummyStrategy()
    if self.params["multi_worker_strat"]:
      name = "multi_worker_mirrored"
    else:
      name = "mirrored"
    strat = distribution_utils.get_distribution_strategy(
        distribution_strategy=name, num_gpus=self.params["num_gpus"])
    return strat

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    opt = optimizer.LazyAdam(
        params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])
    return opt

def _get_log_dir_or_default(flags_obj):
  """Gets init_logdir_timestamp if it is given, otherwise use current time."""
  if flags_obj.init_logdir_timestamp is not None:
    timestamp = flags_obj.init_logdir_timestamp
  else:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
  return os.path.join(flags_obj.model_dir, timestamp)


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)
    if flags_obj.mode == "train":
      task.train()
    elif flags_obj.mode == "predict":
      task.predict()
    elif flags_obj.mode == "eval":
      task.eval()
    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  misc.define_transformer_flags()
  tf.compat.v1.app.run(main)

