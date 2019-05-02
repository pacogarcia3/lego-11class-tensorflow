# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

import csv

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  sess.close()

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def label_image(file_name):
  tf.logging.set_verbosity(tf.logging.ERROR)

  
  #file_name = "tf/eval_images/brick1x1_a.jpg"
  model_file = "tf/tf_files/retrained_graph.pb"
  label_file = "tf/tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  #input_mean = 0
  #input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"

  

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  timeTaken=end-start
  #print("timetaken")
  #print(timeTaken)

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  template = "{} (score={:0.5f})"
  topLabel=""
  prev_result=0
  print("** TensorFlow Result: **")
  for i in top_k:

    print(template.format(labels[i], results[i]))
    if results[i]>prev_result:
      topLabel=labels[i]
    prev_result=results[i]
    
  print("<<<<<<<< "+topLabel+" is the top Result")

  sess.close()
  return topLabel, timeTaken



def batch_label_image(TFpath,imagePath,csvPath):
  tf.logging.set_verbosity(tf.logging.ERROR)

  
  #file_name = "tf/eval_images/brick1x1_a.jpg"
  model_file = TFpath+"/retrained_graph.pb"
  label_file = TFpath+"/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  #input_mean = 0
  #input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"




  if not tf.gfile.IsDirectory(imagePath):
    #tf.logging.fatal('imagePath directory does not exist %s', imagePath)
    print('imagePath directory does not exist %s', imagePath)
    return

  image_list = tf.gfile.ListDirectory(imagePath)
  
  graph = load_graph(model_file)


  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  resultString=""
  with tf.Session(graph=graph) as sess:
    start = time.time()
    for img in image_list:
      if img[-4:]!=".jpg": break
      
      resultString=""
      start_eval = time.time()
      start_t = time.time()
      print(img)
      t = read_tensor_from_image_file(imagePath+"/"+img,
                                input_height=input_height,
                                input_width=input_width,
                                input_mean=input_mean,
                                input_std=input_std)
      end_t=time.time()
      #print('\nTensor time (1-image): {:.3f}s\n'.format(end_t-start_t))
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
      end_eval=time.time()
      #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end_eval-start_eval))

      results = np.squeeze(results)   
      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)
      template = "{} (score={:0.5f})"
      
      topLabel=""
      prev_result=0
      resultString="TensorFlow Result: "
      #print("** TensorFlow Result: **")
      for i in top_k:
        resultString=resultString+" "+template.format(labels[i], results[i])
        #print(template.format(labels[i], results[i]))
        if results[i]>prev_result:
          topLabel=labels[i]
        prev_result=results[i]
      
      writeCSV(csvPath,img,topLabel,resultString) 
      #print("<<<<<<<< "+topLabel+" is the top Result")

    end=time.time()


  timeTaken=end-start
  #print("timetaken")
  #print(timeTaken)

  print('\nTotal time: {:.3f}s\n'.format(end-start))

  sess.close()
  return


def writeCSV(csvPath,i,topLabel,resultString):

  csvfile=csvPath+"\imageRec.csv"
          
  with open(csvfile, 'a+') as csvfile:
    fwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #dimensions:  CountourArea, rotatedWidth,rotatedHeight,minrectArea
    fwriter.writerow([i,topLabel,resultString])
