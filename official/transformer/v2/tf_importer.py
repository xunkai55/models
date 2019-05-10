"""tf_importer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as _tf

try:
  import tensorflow.contrib as _contrib
  _is_tf_v2_globally = False
except ImportError:
  _is_tf_v2_globally = True

if _is_tf_v2_globally:
  import tensorflow as tf_v2
  import tensorflow.compat.v1 as tf_v1
else:
  import tensorflow as tf_v1
  import tensorflow.compat.v2 as tf_v2

# The TF version used in Transformer V2 project.
tf = tf_v2

if tf == tf_v2 and not _is_tf_v2_globally:
  # tf_v1.enable_v2_behavior()
  tf.compat.v1 = tf_v1

