#!/usr/bin/env python3
'''run the model on images json file and produce output
'''

import sys
import argparse
import json
import os
import glob
import cv2
import logging
import tensorflow as tf
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description=\
                                 "run the model on images json file and produce output")

#parser.add_argument('json_dir', nargs='?', type=argparse.FileType('r'),
 #       default=sys.stdin, help='file of records in json format [default: stdin]')
parser.add_argument("subset_name", help="is it train/test/validate dataset?")
    
parser.add_argument('json_dir', nargs='?', type=argparse.FileType('r'),
        default=sys.stdin, help='file of records in json format [default: stdin]')    

parser.add_argument("images_dir", help="input images directory")

parser.add_argument("run_name", help="input run name")

parser.add_argument("output_json_path", help="directory of the folder to store json files",type=str)

parser.add_argument("-b", "--batch", type=int, default=None,
        help="set batch size [default: size is 32]")

parser.add_argument('-r', '--resize', default=None,
        help="resize image e.g. 256x256 [default: do nothing]")

args = parser.parse_args()


# Control memory usage

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


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


if args.resize:
    image_size = tuple(map(int, args.resize.split('x')))
    logging.info("resizing to {}x{}".format(*image_size))
else:
    image_size = None
    logging.info("not resizing images")
    
if args.batch:
        batch_size = args.batch
        logging.info("changing batch size to {}".format(batch_size))
else:
    batch_size = 32
    logging.info("batch size remains 32")

json_filenames = []
records = json.load(args.json_dir)
filenames2labels = defaultdict(list)



for r in records:
    json_filenames.append(r['filename'])
    filenames2labels[r['filename']].append(r['labels'])


'''    
for image_path in glob.glob(os.path.join(args.images_dir,'*')):
    filename = image_path[-16:]
    print(filename)
    if(filename in json_filenames):
        filenames.append(filename)
        labels.append(filenames2labels[filename][0])
        images.append(_load_image(image_path, image_size))
#images_json = json.load(args.images_json)

for record in images_json:
     try:
        image = _load_image(record['image'])
        data.append(image)
        
    except (OSError, ValueError):
        # file doesn't exist in this directory
        continue'''

def _augment(filename, image, labels):
    '''Placeholder for data augmentation
    '''
    image = tf.reshape(image, [256, 256, 1])
    return filename, image, labels


def _normalize(filename, image, labels):
    '''Convert `image` from [0, 255] -> [-0.5, 0.5] floats
    '''
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return filename, image, labels

def inputs(filenames, images, labels, batch_size, num_epochs, size = image_size):
    ''' Reads input data num_epochs times or forever if num_epochs is None
        returns dataset, iterator pair
    '''

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
           
        #dataset = tf.data.TFRecordDataset(filenames)
    
        def gen():
            for i,_ in enumerate(filenames):
                yield(filenames[i], images[i], labels[i])
                
        dataset = tf.data.Dataset.from_generator(gen,
                                                 (tf.string, tf.float32, tf.int32),
                                                 (tf.TensorShape([]), tf.TensorShape([size[0],size[1]]), tf.TensorShape([len(labels[0])])))
            
        #dataset = tf.data.Dataset.from_generator(lambda: images, tf.float32, tf.TensorShape([256,256]))
        
        # The map transformation takes a function and applies it to every element
        # of the dataset.
        
        #dataset = dataset.map(_decode)
        #dataset = dataset.shard(num_shards, shard_index)
        #dataset = dataset.filter(_filter)
        dataset = dataset.map(_augment)
        dataset = dataset.map(_normalize)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

    return dataset, iterator

def gen_documents(filenames, images, labels, RUN_NAME, batch_size, image_size):
    
    LOG_ROOT = '/var/logs'
    SUMMARY_DIR = os.path.join(LOG_ROOT, 'logs', RUN_NAME)
    MODEL_DIR = os.path.join(LOG_ROOT, 'models', RUN_NAME)
    MODEL_GRAPH = os.path.join(MODEL_DIR, 'vae.meta')
    dir_name = SUMMARY_DIR + '/train'
    
    with tf.Graph().as_default() as graph:

        # Repeatable results
        tf.set_random_seed(0)

        # Get Data
        dataset, iterator = inputs(filenames = filenames, images = images, labels = labels, 
                                   batch_size = batch_size, num_epochs = 1, size = image_size)
        # Log output for Tensorboard
        summary_logger = tf.summary.FileWriter(dir_name, flush_secs=30)

        # Run
        with tf.Session(config=config) as session:

            # restore
            reader = tf.train.import_meta_graph(MODEL_GRAPH)
            reader.restore(session, tf.train.latest_checkpoint(MODEL_DIR))

            # get references to graph endpoints
            filename = tf.get_collection('filename')[0]
            #image = tf.get_collection('image')[0]
            labels = tf.get_collection('labels')[0]
            pred_labels = tf.get_collection('pred_labels')[0]
            
            mu = tf.get_collection('mu')[0]
            sigma = tf.get_collection('sigma')[0]
            #xhat = tf.get_collection('xhat')[0]

            merged = tf.get_collection('merged')[0]

            data_handle = tf.get_collection('data_handle')[0]
            train1 = tf.get_collection('train1')[0]
            train2 = tf.get_collection('train2')[0]
            handle = session.run(iterator.string_handle())
            
            step = 0
            while True:
                try:
                    mu_, sigma_, filename_, labels_, pred_labels_, summary = \
                        session.run([mu, sigma, filename, labels, pred_labels, merged],
                                feed_dict = { data_handle: handle, train1: 0 , train2: 0})

                    name = filename_[0].decode('ascii')
                    json_item = {
                    'filename': name,
                    'subset': args.subset_name,
                    'labels': labels_[0].tolist(), 
                    'pred_labels': pred_labels_[0].tolist(), 
                    'mu' : mu_[0].tolist(),
                    'sigma': sigma_[0].tolist(),
                    #'xhat': (xhat_[0].reshape(256, 256)+0.5).tolist(),    
                    #'image': (image_[0].reshape(256, 256)+0.5).tolist()
                    }
                    
                    if(args.subset_name == 'golden_src'):
                        JSON_DIR = os.path.join(args.output_json_path, name[:12] + "_src.json")
                    elif(args.subset_name == 'golden_dr'):
                        JSON_DIR = os.path.join(args.output_json_path, name[:12] + "_dr.json")
                    else:
                        JSON_DIR = os.path.join(args.output_json_path, name[:12] + ".json")
                    
                    with open(JSON_DIR, 'w') as outfile:
                        json.dump(json_item, outfile, separators=(',', ':'), indent = 2)

                    step += 1
                    summary_logger.add_summary(summary, step)                        
                    print("{}.".format(step), end="", flush=True)

                except tf.errors.OutOfRangeError:
                    print(".done")
                    break
           
        

i = 0        
generate_size = 5000
images = []
filenames = []
labels = []
    
    
#feed images every "generate_size" images    
for i,filename in enumerate(json_filenames):
    image_path = (os.path.join(args.images_dir,filename))
    image = _load_image(image_path, image_size)
    filenames.append(filename)
    labels.append(filenames2labels[filename][0])
    images.append(image)
    print(filename)
    if((i+1) % generate_size == 0):
        gen_documents(filenames, images, labels, args.run_name, 1, image_size)   
        images = []
        filenames = []
        labels = []

#generate the remainder jsons
if((i+1) % generate_size != 0):
    gen_documents(filenames, images, labels, args.run_name, 1, image_size)        
    
print("done writing the output json files")

     

logging.info("done writing the output json file")
sys.exit(0)     