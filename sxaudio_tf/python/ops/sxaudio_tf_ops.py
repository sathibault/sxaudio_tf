"""Use sxaudio_tf ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

sxaudio_tf_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_sxaudio_tf_ops.so'))
spectrogram = sxaudio_tf_ops.spectrogram
