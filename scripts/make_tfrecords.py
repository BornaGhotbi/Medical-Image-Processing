''' make tfrecords from json labels file
'''

import sys
import os
import argparse
import json

import logging
logging.basicConfig(
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    format="%(asctime)s [make_tfrecords] %(levelname)-8s %(message)s",
    level=logging.INFO
)


## Parse command line args

parser = argparse.ArgumentParser(description=\
        "make tfrecords from json labels file")

parser.add_argument('image_dir', help="image directory path")
parser.add_argument('labeled_json', type=argparse.FileType('r'),
        help="input json file")
parser.add_argument('tfrecords', type=str, help="output tf records filename")

parser.add_argument('-r', '--resize', default=None,
        help="resize image e.g. 256x256 [default: do nothing]")

args = parser.parse_args()


## Begin processing

import numpy as np
logging.info("using numpy version {}".format(np.__version__))

import cv2
logging.info("using OpenCV version {}".format(cv2.__version__))

import tensorflow as tf
logging.info("using TensorFlow version {}".format(tf.__version__))


## Helper functions

def _int64_scalar_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_vector_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _load_image(path, size):
    ''' load grayscale image from file; resize if necessary
    '''
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("empty image: {}".format(path))

    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32)
    return img


## Iterate through all labels: images that appear in the directory
##    are added to the tfrecords file

try:
    nb_images = len(os.listdir(args.image_dir))
    logging.info("processing {} images".format(nb_images))

except OSError:
    logging.fatal("image directory does not exist: {}".format(args.image_dir))
    sys.exit(1)


labeled_records = json.load(args.labeled_json)

if args.resize:
    size = tuple(map(int, args.resize.split('x')))
    logging.info("resizing to {}x{}".format(*size))
else:
    size = None
    logging.info("not resizing images")


# open the TFRecords file
with tf.python_io.TFRecordWriter(args.tfrecords) as writer:

    # count the images for user feedback
    img_counter = 0
    for record in labeled_records:

        try:
            image = _load_image(os.path.join(args.image_dir, record['filename']), size)

        except (OSError, ValueError):
            # file doesn't exist in this directory
            continue

        #feature = {
            #'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
        #}

        # TODO: this produces corrupt protobuf, unreadable by our dataset reader; Why?
        feature = {
            'filename': _bytes_feature(tf.compat.as_bytes(record['filename'])),
            'labels': _int64_vector_feature(record['labels']),
            'age': _int64_scalar_feature(int(record['age'][:3])),
            'gender': _bytes_feature(tf.compat.as_bytes(record['gender'])),
            'view':  _bytes_feature(tf.compat.as_bytes(record['view'])),
            'image': _bytes_feature(tf.compat.as_bytes(image.tostring()))
        }

        # create the entire labeled sample
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # output
        writer.write(example.SerializeToString())

        # user feedback
        img_counter += 1
        if not img_counter%100:
            logging.info("processing... {} images".format(img_counter))

logging.info("finished writing {} images".format(img_counter))
sys.exit(0)
