''' make train and test sets from our index of good images
'''

import sys
import json
import argparse

import numpy as np

import logging
logging.basicConfig(
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    format="%(asctime)s [make_train_test] %(levelname)-8s %(message)s",
    level=logging.INFO
)


## Parse command line arguments

class RangeCheckAction(argparse.Action):
    lo, hi = 0, 100

    def __call__(self, parser, namespace, values, option_string=None):
        if values < self.lo or values > self.hi:
            parser.error("Value {} out of range [{}-{}]".format(
                option_string, self.lo, self.hi))
        setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser(description=\
        'generate train and test sets from index of good images')

parser.add_argument('all_json', nargs='?', type=argparse.FileType('r'),
        default=sys.stdin, help='file of all records in json format [default: stdin]')
parser.add_argument('train_json', type=argparse.FileType('w'),
        help='training records')
parser.add_argument('validate_json', type=argparse.FileType('w'),
        help='validation records')
parser.add_argument('test_json', type=argparse.FileType('w'),
        help='testing records')

parser.add_argument('-s', '--seed', type=int, default=42,
        help='random seed [default: 42]')
parser.add_argument('-v', '--validate-percent', action=RangeCheckAction, 
        type=int, default=15, 
        help='percent [0-100] of images to use for validate [default: 15]')
parser.add_argument('-t', '--test-percent', action=RangeCheckAction, 
        type=int, default=20, 
        help='percent [0-100] of images to use for testing [default: 20]')

args = parser.parse_args()


## Build train/validate/test sets

random_state = np.random.RandomState(args.seed)


## Check inputs

train_percent = 100 - args.validate_percent - args.test_percent

if train_percent<=0:
    logging.fatal("no data for training! check validate, test percentages")
    sys.exit(1)
elif train_percent<50:
    logging.warning("only {:.1f}% of data available for training".format(train_percent))
else:
    logging.info("targeting train/validate/test percent = {}/{}/{}".format(
        train_percent, args.validate_percent, args.test_percent))


## Read through all records and split them up

test_threshold = args.test_percent/100.0
validate_threshold = test_threshold + args.validate_percent/100.0

all_records = json.load(args.all_json)
train, validate, test = [], [], []

start = 0
end = 1
records_size = len(all_records)

while end < records_size:
    
    if(all_records[start]['labels'] != [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]):
        
        #keep the filenames with the same number ("00000003") in the same cluster
        #and divide data based on these clusters to train, validation, and test data
        while all_records[end]['filename'][:8] == all_records[start]['filename'][:8] and end < records_size-1:

            end += 1

        _u = random_state.rand()

        if _u < test_threshold:
            test.extend(all_records[start:end])
        elif _u < validate_threshold:
            validate.extend(all_records[start:end])
        else:
            train.extend(all_records[start:end])

        start = end
        end += 1

## Record results

json.dump(train, args.train_json, separators=(',', ':'))
json.dump(validate, args.validate_json, separators=(',', ':'))
json.dump(test, args.test_json, separators=(',', ':'))


## Write stats

logging.info("wrote train/validate/test count = {}/{}/{}".format(
    len(train), len(validate), len(test)))

actual_total = sum([len(train), len(validate), len(test)])
actual_pct_train = 100*len(train)/actual_total
actual_pct_validate = 100*len(validate)/actual_total
actual_pct_test = 100*len(test)/actual_total

logging.info("wrote train/validate/test percent = {:.1f}/{:.1f}/{:.1f}".format(
    actual_pct_train, actual_pct_validate, actual_pct_test))

sys.exit(0)
