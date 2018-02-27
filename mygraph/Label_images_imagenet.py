# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Simple image classification with Inception.

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import cv2
import tensorflow as tf
import numpy as np

cap = cv2.VideoCapture(0)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=True,
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph( labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    running=True
    while running:
        ret, image_np = cap.read()
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)


        ret, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        rgbimg=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        #rgbimg = cv2.flip(rgbimg,1)
        # Numpy array
        np_image_data = np.asarray(rgbimg)[:, :, 0:3]
        #np_final = np.expand_dims(np_image_data, axis=0)

        #print("camera capture!")
        #font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            running=False
            cv2.destroyAllWindows()
        #if cv2.waitKey(25) & 0xFF == ord('c'):
        #print("camera label !")
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        #predictions, = sess.run(softmax_tensor, {'Mul:0': np_final})
        predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': np_image_data})
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 50)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        i=0
        for node_id in top_k:
            i=i+1
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%d node_id' % (node_id))
            if(score>0.3):
                cv2.putText(image_np, '%s (score = %.5f)' % (human_string, score),
                            (10, 50*i),
                            font,
                            fontScale,
                            fontColor,
                            lineType)



        cv2.imshow('object detection', image_np)
    return 0


def main(argv):
  """Runs inference on we'b'camera."""
  if argv[1:]:
      raise ValueError('Unused Command Line Args: %s' % argv[1:])


  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)



  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  run_graph( labels, FLAGS.input_layer, FLAGS.output_layer,
            FLAGS.num_top_predictions)


if __name__ == '__main__':
   FLAGS, unparsed = parser.parse_known_args()
   tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
