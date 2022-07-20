import argparse
import os,sys
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import re
import csv
from collections import OrderedDict
import os
from Common import loss_utils
from Common import pc_util
from Common.pc_util import load, save_ply_property,get_pairwise_distance
from Common.ops import normalize_point_cloud
from Common.utils import AverageMeter

sys.path.append(os.path.join(os.getcwd(),"tf_ops/sampling"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/nn_distance"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/approxmatch"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/grouping"))
import tf_nndistance
from sklearn.neighbors import NearestNeighbors
import math
from time import time
# parser = argparse.ArgumentParser()
# parser.add_argument("--pred", type=str, required=True, help=".xyz")
# parser.add_argument("--gt", type=str, required=True, help=".xyz")
# FLAGS = parser.parse_args()
# PRED_DIR = os.path.abspath(FLAGS.pred)
# GT_DIR = os.path.abspath(FLAGS.gt)
# print(PRED_DIR)
# NAME = FLAGS.name

#print(GT_DIR)
root = os.path.join("/home/lirh/pointcloud/InvertPoint_tf/experiments/new")

num_point = 2048
pred_tensor = tf.placeholder(tf.float32, [1, num_point, 3])
gt_tensor = tf.placeholder(tf.float32, [1, num_point, 3])
if False:
    pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred_tensor)
    gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt_tensor)

cd_forward, _, cd_backward, _ = tf_nndistance.nn_distance(pred_tensor, gt_tensor)
cd_forward = cd_forward[0, :]
cd_backward = cd_backward[0, :]

emd_dis = loss_utils.earth_mover(pred_tensor, gt_tensor)
_, shapeDis, den8, den16, den24 = loss_utils.get_Geometric_Loss(pred_tensor, gt_tensor, return_all=True)


datasets = ["model40"][0]
modes = ["uniform","random","partial","scan"][:]
subfixs =  ["INV_20201101-1127"]  # 512



#subfixs =  ["INV_20201121-1135"]  # 256
#subfixs =  ["INV_20201121-1136"]  # 128
with tf.Session() as sess:

    fieldnames = ["name", "CD", "CD_F", "CD_B", "HD", "EMD", "MD", "Den8","Den16","Den24"]#,"Den2","Den3"]
    for dataset in datasets:
        for subfix in subfixs:
            for mode in modes:
                gt_paths = glob(os.path.join(root, '%s/HD/%s/*.xyz' % (dataset, mode)))
                print(subfix,"-------",mode, "------", dataset, "------", len(gt_paths))
                avg_cd_forward_value = AverageMeter()
                avg_cd_backward_value = AverageMeter()
                avg_hd_value = AverageMeter()
                avg_emd_value = AverageMeter()
                avg_md_value = AverageMeter()
                avg_den8 = AverageMeter()
                avg_den16 = AverageMeter()
                avg_den24 = AverageMeter()


                counter = 0

                source_dir = os.path.join(root,dataset,subfix,mode)
                print("evaluate_folder:",source_dir)
                csv_file = os.path.join(source_dir,"evaluation_%s_%s.csv"%(subfix,mode))
                with open(csv_file, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
                    writer.writeheader()

                    for gt_path in tqdm(gt_paths, total=len(gt_paths)):
                        row = {}
                        gt = load(gt_path)[:, :3]
                        gt = gt[np.newaxis, ...]

                        pred_path = os.path.basename(gt_path).replace("_HD_","_%s_"%subfix)
                        pred = pc_util.load(os.path.join(source_dir,pred_path))
                        pred = pred[:, :3]

                        row["name"] = os.path.basename(pred_path)
                        pred = pred[np.newaxis, ...]
                        cd_forward_value, cd_backward_value, emd_val, shape_val, den8_val, den16_val, den24_val = \
                                                            sess.run([cd_forward, cd_backward, emd_dis, shapeDis,
                                                                      den8, den16, den24],
                                                                     feed_dict={pred_tensor:pred, gt_tensor:gt})

                        #save_ply_property(np.squeeze(pred), cd_forward_value, pred_path[:-4]+"_cdF.ply", property_max=0.003, cmap_name="jet")
                        #save_ply_property(np.squeeze(gt), cd_backward_value, pred_path[:-4]+"_cdB.ply", property_max=0.003, cmap_name="jet")
                        md_value = np.mean(cd_forward_value)+np.mean(cd_backward_value)
                        hd_value = np.max(np.amax(cd_forward_value, axis=0)+np.amax(cd_backward_value, axis=0))
                        cd_backward_value = np.mean(cd_backward_value)
                        cd_forward_value = np.mean(cd_forward_value)
                        row["CD"] = cd_forward_value+cd_backward_value
                        row["CD_F"] = cd_forward_value
                        row["CD_B"] = cd_backward_value
                        row["HD"] = hd_value
                        row["EMD"] = emd_val
                        row["MD"] = shape_val
                        row["Den8"] = den8_val
                        row["Den16"] = den16_val
                        row["Den24"] = den24_val

                        avg_cd_forward_value.update(cd_forward_value)
                        avg_cd_backward_value.update(cd_backward_value)
                        avg_hd_value.update(hd_value)
                        avg_emd_value.update(emd_val)
                        avg_md_value.update(shape_val)
                        avg_den8.update(den8_val)
                        avg_den16.update(den16_val)
                        avg_den24.update(den24_val)



                        writer.writerow(row)
                        counter += 1

                    row = OrderedDict()

                    #avg_md_forward_value /= counter
                    #avg_md_backward_value /= counter
                    #avg_hd_value /= counter
                    #avg_emd_value /= counter
                    row["CD"] = avg_cd_forward_value.avg + avg_cd_backward_value.avg
                    row["CD_F"] = avg_cd_forward_value.avg
                    row["CD_B"] = avg_cd_backward_value.avg
                    row["HD"] = avg_hd_value.avg
                    row["EMD"] = avg_emd_value.avg
                    row["MD"] = avg_md_value.avg
                    row["Den8"] = avg_den8.avg
                    row["Den16"] = avg_den16.avg
                    row["Den24"] = avg_den24.avg

                    print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))

                    writer.writerow(row)
                    print("|".join(["{:>15.8f}".format(d) for d in row.values()]))
                    print("out_file", csv_file)

