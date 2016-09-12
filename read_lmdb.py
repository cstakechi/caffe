import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

lmdb_env = lmdb.open('/home/caffe2/Caffe/examples/mwam/mwam_test_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)
    print str(key) +':'+str(label)
