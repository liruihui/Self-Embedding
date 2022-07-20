from __future__ import print_function
import os,sys
#sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from Invertible.model_inv import Model

from Invertible.configs_inv import FLAGS
from datetime import datetime
import logging
import pprint
pp = pprint.PrettyPrinter()



def run():
    if not FLAGS.restore:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, current_time)
        try:
            os.makedirs(FLAGS.log_dir)
        except os.error:
            pass
    print('checkpoints:', FLAGS.log_dir)
    print('data_dir:', FLAGS.data_dir)

    pp.pprint(FLAGS)
    model = Model(FLAGS)
    model.train()

def main(unused_argv):
  run()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
