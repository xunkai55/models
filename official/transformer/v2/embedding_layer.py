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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.transformer.v2 import tf_importer
tf = tf_importer.tf


class EmbeddingSharedWeights(tf.keras.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, method="gather"):
    """Specify characteristic parameters of embedding layer.

    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indices and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    if method not in ("gather", "matmul"):
      raise ValueError("method {} must be 'gather' or 'matmul'".format(method))
    self.method = method

    with tf.name_scope("embedding_and_softmax"):
      # Create and initialize weights. The random normal initializer was chosen
      # randomly, and works well.
      self.shared_weights = self.add_weight(
          "weights",
          shape=[self.vocab_size, self.hidden_size],
          dtype="float32",
          initializer=tf.random_normal_initializer(
              mean=0., stddev=self.hidden_size**-0.5))

  def get_config(self):
    return {
        "vocab_size": self.vocab_size,
        "hidden_size": self.hidden_size,
        "method": self.method,
    }

  def call(self, x):
    """Get token embeddings of x.

    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
    """
    with tf.name_scope("embedding"):
      # Create binary mask of size [batch_size, length]
      mask = tf.cast(tf.not_equal(x, 0), tf.float32)

      if self.method == "gather":
        embeddings = tf.gather(self.shared_weights, x)
        embeddings *= tf.expand_dims(mask, -1)
      else:  # matmul
        raise ValueError("Not supported method: {}".format(self.method))

      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size ** 0.5

      return embeddings

  def linear(self, x):
    """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      x = tf.reshape(x, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])

