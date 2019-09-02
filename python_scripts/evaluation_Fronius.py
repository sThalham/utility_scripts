#!/usr/bin/env python

import os
import numpy as np
import transforms3d as tf3d
import copy
import open3d

from scipy import ndimage, signal
import argparse
import sys
import math
import cv2

import keras
import tensorflow as tf
import open3d

# Allow relative imports when being executed as script.
#if __name__ == "__main__" and __package__ is None:
#    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#    import RetNetPose.bin  # noqa: F401
#    __package__ = "RetNetPose.bin"
sys.path.append("/home/sthalham/workspace/RetNetPose")
from RetNetPose import models
from RetNetPose.utils.config import read_config_file, parse_anchor_parameters
from RetNetPose.utils.eval import evaluate
from RetNetPose.utils.keras_version import check_keras_version

# mAP
# Precision = True positive / (True positive + False positive)
# Recall = True positive / (True positive + False negative)

#fxkin = 579.68  # blender calculated
#fykin = 542.31  # blender calculated
#cxkin = 320
#cykin = 240

fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899

threeD_boxes = np.ndarray((4, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[0.05, 0.04, 0.03],  # AC-Abdeckung [81, 68, 25] expand last dim
                                     [0.05, 0.04, -0.03],
                                     [0.05, -0.04, -0.03],
                                     [0.05, -0.04, 0.03],
                                     [-0.05, 0.04, 0.03],
                                     [-0.05, 0.04, -0.03],
                                     [-0.05, -0.04, -0.03],
                                     [-0.05, -0.04, 0.03]])
threeD_boxes[1, :, :] = np.array([[0.05, 0.04, 0.03],  # Deckel [81, 72, 38]
                                     [0.05, 0.04, -0.03],
                                     [0.05, -0.04, -0.03],
                                     [0.05, -0.04, 0.03],
                                     [-0.05, 0.04, 0.03],
                                     [-0.05, 0.04, -0.03],
                                     [-0.05, -0.04, -0.03],
                                     [-0.05, -0.04, 0.03]])
threeD_boxes[2, :, :] = np.array([[0.060, 0.1, 0.03],  # Seite links [120, 198, 45]
                                     [0.060, 0.1, -0.03],
                                     [0.060, -0.1, -0.03],
                                     [0.060, -0.1, 0.03],
                                     [-0.060, 0.1, 0.03],
                                     [-0.060, 0.1, -0.03],
                                     [-0.060, -0.1, -0.03],
                                     [-0.060, -0.1, 0.03]])
threeD_boxes[3, :, :] = np.array([[0.060, 0.1, 0.03], # Seite links [120, 206, 56]
                                  [0.060, 0.1, -0.03],
                                  [0.060, -0.1, -0.03],
                                  [0.060, -0.1, 0.03],
                                  [-0.060, 0.1, 0.03],
                                  [-0.060, 0.1, -0.03],
                                  [-0.060, -0.1, -0.03],
                                  [-0.060, -0.1, 0.03]])


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):

    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=640)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def load_model(args=None):
    check_keras_version()

    anchor_params = None
    backbone = 'resnet50'
    score_threshold = 0.05
    model = '/home/sthalham/data/MMAssist_Fronius/RetNetPose_weights/resnet50_linemod_50.h5'

    #if args.config and 'anchor_parameters' in args.config:
    #    anchor_params = parse_anchor_parameters(args.config)

    print('Loading model, this may take a second...')
    model = models.load_model(model, backbone_name=backbone)
    graph = tf.get_default_graph()
    model = models.convert_model(model, anchor_params=anchor_params) # convert model

    # print model summary
    print(model.summary())

    return model, score_threshold, graph


def create_point_cloud(depth, fx, fy, cx, cy, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud_final = np.transpose(np.array((xP, yP, zP)))
    cloud_final[cloud_final[:,2]==0] = np.NaN


    return cloud_final


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


if __name__ == "__main__":

    root = '/home/sthalham/workspace/RetNetPose/'
    jsons = root + 'val_bbox_results.json'
    model_path = "/home/sthalham/data/LINEMOD/models/"

    model, score_threshold, graph = load_model()

    visu = True

    model_radii = np.array([0.060, 0.064, 0.0121, 0.0127])
    objectNames = ['AC_Abdeckung', 'Deckel', 'Seite_links', 'Seite_rechts']

    ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender/AC_Abdeckung.ply'
    pcd_model_1 = open3d.read_point_cloud(ply_path)
    open3d.estimate_normals(pcd_model_1, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender/Deckel.ply'
    pcd_model_2 = open3d.read_point_cloud(ply_path)
    open3d.estimate_normals(pcd_model_2, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender/Seitenteil_links.ply'
    pcd_model_3 = open3d.read_point_cloud(ply_path)
    open3d.estimate_normals(pcd_model_3, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender/Seitenteil_rechts.ply'
    pcd_model_4 = open3d.read_point_cloud(ply_path)
    open3d.estimate_normals(pcd_model_4, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    testData = '/home/sthalham/data/MMAssist_Fronius/test_daten/tests032019/today1-img-depth/'
    sub = os.listdir(testData)

    for s in sub:

        dep_name = testData + s
        image_dep = cv2.imread(dep_name, cv2.IMREAD_UNCHANGED)
        image_dep = cv2.resize(image_dep, (640, 480), None, fxkin, fykin, cv2.INTER_NEAREST)

        # inpainting
        #scaleOri = np.amax(image_dep)
        #inPaiMa = np.where(image_dep == 0.0, 255, 0)
        #inPaiMa = inPaiMa.astype(np.uint8)
        #inPaiDia = 3.0
        #depth_refine = image_dep.astype(np.float32)
        #depPaint = cv2.inpaint(image_dep, inPaiMa, inPaiDia, cv2.INPAINT_NS)
        #depNorm = depPaint - np.amin(depPaint)
        #rangeD = np.amax(depNorm)
        #depNorm = np.divide(depNorm, rangeD)
        #image_dep = np.multiply(depNorm, scaleOri)

        image_dep[image_dep > 1500.0] = 0
        scaCro = 255.0 / np.nanmax(image_dep)
        cross = np.multiply(image_dep, scaCro)
        image = cross.astype(np.uint8)
        image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

            # print(image.shape)
        with graph.as_default():
            boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        det1 = False
        det2 = False
        det3 = False
        det4 = False

        print(scores)

        # compute predicted labels and scores
        for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
            # scores are sorted, so we can break

            if score < score_threshold:
                continue

            if label < 0:
                continue
            elif label == 0 and det1 == False:
                det1 = True
            elif label == 1 and det2 == False:
                det2 = True
            elif label == 2 and det3 == False:
                det3 = True
            elif label == 3 and det4 == False:
                det4 = True
            else:
                continue

            control_points = box3D[(label), :]

            dC = label

            print('--------------------- BBox center as initial estimate -------------------')
            dep_val = image_dep[int(box[1] + (box[3] * 0.5)), int(box[0] + (box[2] * 0.5))]
            dep = dep_val + model_radii[dC - 1]

            x_o = (((box[0] + (box[2] * 0.5)) - cxkin) * dep) / fxkin
            y_o = (((box[1] + (box[3] * 0.5)) - cykin) * dep) / fykin

            irvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            itvec = np.array([x_o, y_o, dep], dtype=np.float32)

            print('--------------------- PnP Pose Estimation -------------------')
            obj_points = np.ascontiguousarray(threeD_boxes[dC - 1, :, :], dtype=np.float32)
            est_points = np.ascontiguousarray(np.asarray(control_points, dtype=np.float32).T,
                                                  dtype=np.float32).reshape(
                    (8, 1, 2))

            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
            retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                   imagePoints=est_points, cameraMatrix=K,
                                                                   distCoeffs=None, rvec=irvec, tvec=itvec,
                                                                   useExtrinsicGuess=True, iterationsCount=100,
                                                                   reprojectionError=8.0, confidence=0.99,
                                                                   flags=cv2.SOLVEPNP_ITERATIVE)
            rmat, _ = cv2.Rodrigues(orvec)
            # rd = re(np.array(gtRots[j], dtype=np.float32).reshape(3, 3), rmat)
            # xyz = te((np.array(gtPoses[j], dtype=np.float32)*0.001), (otvec.T))

            print('--------------------- ICP refinement -------------------')
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])),
                                            (0, 0, 0), 3)


            pcd_img = create_point_cloud(image_dep, fxkin, fykin, cxkin, cykin, 0.001)
            pcd_img = pcd_img.reshape((480, 640, 3))[int(box[1]):int(box[1] + box[3]),
                          int(box[0]):int(box[0] + box[2]), :]
            pcd_img = pcd_img.reshape((pcd_img.shape[0] * pcd_img.shape[1], 3))
            pcd_crop = open3d.PointCloud()
            open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(
                    radius=0.02, max_nn=30))
            #open3d.draw_geometries([pcd_crop])

            guess = np.zeros((4, 4), dtype=np.float32)
            guess[:3, :3] = rmat
            guess[:3, 3] = itvec.T
            guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

            if dC == 0:
                pcd_model = pcd_model_1
            elif dC == 1:
                pcd_model = pcd_model_2
            elif dC == 2:
                pcd_model = pcd_model_3
            elif dC == 3:
                pcd_model = pcd_model_4
            reg_p2p = open3d.registration_icp(pcd_model, pcd_crop, 0.015, guess,
                                                  open3d.TransformationEstimationPointToPoint())
            R_est = reg_p2p.transformation[:3, :3]
            t_est = reg_p2p.transformation[:3, 3]

            # Visualize Detections

            # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])),
            #                 (0, 0, 0), 3)

            font = cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCornerOfText = (int(box[0]) + 5, int(box[1]) + int(box[3]) - 5)
            fontScale = 0.5
            fontColor = (25, 215, 250)
            fontthickness = 2
            lineType = 2
            gtText = objectNames[dC - 1]

            fontColor2 = (0, 0, 0)
            fontthickness2 = 4
            cv2.putText(image, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor2,
                            fontthickness2,
                            lineType)

            cv2.putText(image, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            fontthickness,
                            lineType)

            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
            axisPoints, _ = cv2.projectPoints(points, R_est, t_est * 1000.0, K, (0, 0, 0, 0))

            image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
            image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
            image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

            cv2.imwrite('/home/sthalham/visTests/fro_test.jpg', image)
            print('stop')



