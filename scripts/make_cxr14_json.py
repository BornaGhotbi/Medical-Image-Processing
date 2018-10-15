''' make json labels file from the CXR14 labels csv
'''

import sys
import os
import argparse
import json

from collections import OrderedDict

import logging
logging.basicConfig(
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    format="%(asctime)s [make_cxr14_json] %(levelname)-8s %(message)s",
    level=logging.INFO
)


## Parse command line args

parser = argparse.ArgumentParser(description=\
        "make json labels file from the CXR14 labels csv")

parser.add_argument('labels_csv', nargs='?', type=argparse.FileType('r'),
        default=sys.stdin, help="input csv file")
parser.add_argument('labels_json', nargs='?', type=argparse.FileType('w'),
        default=sys.stdout, help="output json file")
parser.add_argument('-p', '--print-keys', action="store_true",
        help="print label key lookup and exit")
parser.add_argument('-i', '--indent', type=int, default=None,
        help="format output with indentation [default: minimize output]")

args = parser.parse_args()


## Define label keys
LABEL_KEYS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "No Finding",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
nb_label_keys = len(LABEL_KEYS)
label_keys_by_id = LABEL_KEYS
ids_by_label_key = OrderedDict([(v, k) for k, v in enumerate(label_keys_by_id)])

VIEW_KEYS = [
    "PA",
    "AP"
]
view_keys_by_id = VIEW_KEYS
ids_by_view_key = OrderedDict([("PA", 0), ("AP", 1)])


## Optionally print out the label keys and exit
if args.print_keys:
    print(json.dumps({
            "labels": ids_by_label_key,
            "views": ids_by_view_key
        }, separators=(',', ':'), indent=2))
    sys.exit(0)


## Accumulate labels in a python list of dicts
all_labels = []

# Eat header
args.labels_csv.readline()

# Iterate through file; represent labels as N-hot binary lists
line_no = 2
for line in args.labels_csv.readlines():

    try:
        tokens = line.strip().split(',')
        labels = [0]*nb_label_keys

        for key in tokens[1].split('|'):
            labels[ids_by_label_key[key]] = 1

        all_labels.append({
            'filename': tokens[0],
            'labels': labels,
            'age':tokens[4],
            'gender': tokens[5],
            'view': tokens[6],
            'dx': float(tokens[9]),
            'dy': float(tokens[10]),
            'w': int(tokens[7]),
            'h': int(tokens[8]),
        })

    except ValueError:
        logging.warning("skipping bad line: {}".format(line_no))

    finally:
        line_no += 1

# Write file
json.dump(all_labels, args.labels_json, separators=(',', ':'), indent=args.indent)

logging.info("done reading {} lines".format(line_no))
sys.exit(0)
