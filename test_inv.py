from __future__ import print_function
import os,sys
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
#from Invertible.model_inv_major import Model
from Invertible.model_inv import Model
from Invertible.configs_inv import FLAGS
from datetime import datetime
import logging
import pprint
pp = pprint.PrettyPrinter()



def run():
    FLAGS.batch_size=1

    mode_type = "20201101-1127" # random
    #mode_type = "20211206-1901" # partial

    FLAGS.log_dir = "log/"+mode_type
    FLAGS.restore_epoch = 400

    # FLAGS.log_dir = "log/20201019-2002" #random train bad
    # FLAGS.log_dir = "log/20201018-1611"
    # FLAGS.restore_epoch = 100
    FLAGS.dataset = ["model40"][0]
    #FLAGS.mode = "uniform"

    print('checkpoints:', FLAGS.log_dir)
    print('data_dir:', FLAGS.data_dir)

    #mode_type = ["random", "uniform", "mix"][0]
    pp.pprint(FLAGS)
    model = Model(FLAGS)
    model.test(mode_type=mode_type)
    model.eval_PU(mode_type=mode_type)

    #model.eval_DiffPoints(mode_type=mode_type)
    #model.eval_large(mode_type=mode_type)


def main(unused_argv):
  run()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
