# Point Set Self-Embedding

This repository contains a Tensorflow implementation of the paper :

[Point Set Self-Embedding](https://ieeexplore.ieee.org/document/9727090/). 
<br>
[Ruihui Li](https://liruihui.github.io/), 
[Xianzhi Li](https://nini-lxz.github.io/),
[Tien-Tsin Wong](https://www.cse.cuhk.edu.hk/~ttwong/), 
[Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/).
<br>
TVCG 2022


### Usage

1. Installation instructions for Ubuntu 16.04:
    * Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. Only this configurations has been tested:
        - Python 3.6.9, TensorFlow 1.11.1
    * Follow <a href="https://www.tensorflow.org/install/pip">Tensorflow installation procedure</a>.

2. Compile the customized TF operators by `sh complile_op.sh`.
   Follow the information from [here](https://github.com/yanx27/PointASNL) to compile the TF operators.

3. Train/Test the model:
   First, you need to download the training data and testing in HDF5 format from [GoogleDrive](https://drive.google.com/drive/folders/1cBwiyUWAHjsiE1cH9Ti1eKmNW8fpdrpD?usp=sharing).
   Then run:
   ```shell
   python train/test_inv.py 
   ```


## Citation

If this paper is useful for your research, please consider citing:

    @article{li2021point,
         title={Point Set Self-Embedding},
         author={Li, Ruihui and Li, Xianzhi and Wong, Tien-Tsin and Fu, Chi-Wing},
         journal={IEEE Transactions on Visualization and Computer Graphics},
         year={2022},
         publisher={IEEE}
     }


### Questions

### Questions

Please contact 'lirh@cse.cuhk.edu.hk'

p.s. This code is kind of messy due to my graduation before.

