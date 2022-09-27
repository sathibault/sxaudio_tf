"""Tests for sxaudio_tf ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_sxaudio_tf.python.ops.sxaudio_tf_ops import sxaudio_tf
except ImportError:
  from sxaudio_tf_ops import sxaudio_tf


class ZeroOutTest(test.TestCase):

  def testZeroOut(self):
    with self.test_session():
      self.assertAllClose(
          sxaudio_tf([[1, 2], [3, 4]]), np.array([[1, 0], [0, 0]]))


if __name__ == '__main__':
  test.main()
