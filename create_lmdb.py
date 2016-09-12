import os

import caffe
import lmdb

from caffe.proto import caffe_pb2
import numpy
import PIL.Image
import random

IMAGE_SIZE = 32

def make_datum(image, label):
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        label=label,
        data=numpy.rollaxis(numpy.asarray(image), 2).tostring())

train_lmdb = '/home/caffe2/Caffe/examples/mwam/mwam_train_lmdb'
test_lmdb = '/home/caffe2/Caffe/examples/mwam/mwam_test_lmdb'

os.system('rm -rf  '+train_lmdb)
os.system('rm -rf  '+test_lmdb)

print 'filepaths'

label_file_map = {}
filepaths = []
for dirpath, _, filenames in os.walk('./jean-ken'):
    for filename in filenames:
        if filename.endswith(('.png', '.jpg')):
            file = os.path.join(dirpath, filename)
            filepaths.append(file)
            label_file_map[file] = 0
            print file
for dirpath, _, filenames in os.walk('./rib'):
    for filename in filenames:
        if filename.endswith(('.png', '.jpg')):
            file = os.path.join(dirpath, filename)
            filepaths.append(file)
            label_file_map[file] = 1
            print file

random.shuffle(filepaths)

print 'train'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, file in enumerate(filepaths):
        image = PIL.Image.open(file)
        datum = make_datum(image, label_file_map[file])
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + file
in_db.close()

print 'test'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, file in enumerate(filepaths):
        image = PIL.Image.open(file)
        datum = make_datum(image, label_file_map[file])
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + file
in_db.close()
