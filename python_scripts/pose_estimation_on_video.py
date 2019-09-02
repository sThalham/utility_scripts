#!/usr/bin/env python

import sys
from scipy import ndimage, signal
import argparse
import os
import sys
import math
import cv2
import numpy as np
import copy
import transforms3d as tf3d

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


fxkin = 575.81573
fykin = 575.81753
cxkin = 314.5
cykin = 235.5

threeD_boxes = np.ndarray((8, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[0.060, 0.1, 0.03],  # Seite-AC [120, 198, 45] id:
                                     [0.060, 0.1, -0.03],
                                     [0.060, -0.1, -0.03],
                                     [0.060, -0.1, 0.03],
                                     [-0.060, 0.1, 0.03],
                                     [-0.060, 0.1, -0.03],
                                     [-0.060, -0.1, -0.03],
                                     [-0.060, -0.1, 0.03]])
threeD_boxes[1, :, :] = np.array([[0.05, 0.04, 0.03],  # AC-Abdeckung [81, 68, 25] expand last dim
                                     [0.05, 0.04, -0.03],
                                     [0.05, -0.04, -0.03],
                                     [0.05, -0.04, 0.03],
                                     [-0.05, 0.04, 0.03],
                                     [-0.05, 0.04, -0.03],
                                     [-0.05, -0.04, -0.03],
                                     [-0.05, -0.04, 0.03]])
threeD_boxes[2, :, :] = np.array([[0.05, 0.04, 0.03],  # DC [81, 72, 38]
                                     [0.05, 0.04, -0.03],
                                     [0.05, -0.04, -0.03],
                                     [0.05, -0.04, 0.03],
                                     [-0.05, 0.04, 0.03],
                                     [-0.05, 0.04, -0.03],
                                     [-0.05, -0.04, -0.03],
                                     [-0.05, -0.04, 0.03]])
threeD_boxes[3, :, :] = np.array([[0.145, 0.06, 0.03],  # Boden [290, 110, 50]
                                     [0.145, 0.06, -0.03],
                                     [0.145, -0.06, -0.03],
                                     [0.145, -0.06, 0.03],
                                     [-0.145, 0.06, 0.03],
                                     [-0.145, 0.06, -0.03],
                                     [-0.145, -0.06, -0.03],
                                     [-0.145, -0.06, 0.03]])
threeD_boxes[4, :, :] = np.array([[0.060, 0.1, 0.03], # Seite-DC [120, 206, 56]
                                  [0.060, 0.1, -0.03],
                                  [0.060, -0.1, -0.03],
                                  [0.060, -0.1, 0.03],
                                  [-0.060, 0.1, 0.03],
                                  [-0.060, 0.1, -0.03],
                                  [-0.060, -0.1, -0.03],
                                  [-0.060, -0.1, 0.03]])
threeD_boxes[5, :, :] = np.array([[0.13, 0.09, 0.07], #Front  [260, 180, 140]
                                  [0.13, 0.09, -0.07],
                                  [0.13, -0.09, -0.07],
                                  [0.13, -0.09, 0.07],
                                  [-0.13, 0.09, 0.07],
                                  [-0.13, 0.09, -0.07],
                                  [-0.13, -0.09, -0.07],
                                  [-0.13, -0.09, 0.07]])
threeD_boxes[6, :, :] = np.array([[0.15, 0.085, 0.05], # Leistungsteil [300, 170, 100]
                                  [0.15, 0.085, -0.05],
                                  [0.15, -0.085, -0.05],
                                  [0.15, -0.085, 0.05],
                                  [-0.15, 0.085, 0.05],
                                  [-0.15, 0.085, -0.05],
                                  [-0.15, -0.085, -0.05],
                                  [-0.15, -0.085, 0.05]])
threeD_boxes[7, :, :] = np.array([[0.13, 0.09, 0.05], # Mantel [260, 180, 100]
                                  [0.13, 0.09, -0.05],
                                  [0.13, -0.09, -0.05],
                                  [0.13, -0.09, 0.05],
                                  [-0.13, 0.09, 0.05],
                                  [-0.13, 0.09, -0.05],
                                  [-0.13, -0.09, -0.05],
                                  [-0.13, -0.09, 0.05]])

threeD_dims = np.ndarray((8, 3), dtype=np.float32)
threeD_dims[0, :] = np.array([0.12, 0.198, 0.045])
threeD_dims[1, :] = np.array([0.081, 0.068, 0.025])
threeD_dims[2, :] = np.array([0.081, 0.072, 0.038])
threeD_dims[3, :] = np.array([0.29, 0.11, 0.05])
threeD_dims[4, :] = np.array([0.12, 0.206, 0.056])
threeD_dims[5, :] = np.array([0.26, 0.18, 0.14])
threeD_dims[6, :] = np.array([0.30, 0.17, 0.10])
threeD_dims[7, :] = np.array([0.26, 0.18, 0.1])

objectNames = ['Seite-AC', 'AC-Abdeckung', 'DC-Abdeckung', 'Boden', 'Seite-DC', 'Front', 'Leistungsteil', 'Mantel']
# fronius_6DoF: Dc-Abdeckung=3, Front=6

ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Seite_AC.ply'
pcd_model_1 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/AC_Abdeckung.ply'
pcd_model_2 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/DC_Abdeckung.ply'
pcd_model_3 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Boden.ply'
pcd_model_4 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Seite_DC.ply'
pcd_model_5 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Front.ply'
pcd_model_6 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Leistungsteil.ply'
pcd_model_7 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Mantel.ply'
pcd_model_8 = open3d.read_point_cloud(ply_path)


def make_video(images, name, fps=25, size=None,
               is_color=True, format="XVID"):
    fourcc = cv2.VideoWriter_fourcc(*format)
    # fourcc = cv.cv.CV_FOURCC(*'XVID')
    vid = cv2.VideoWriter(name, fourcc, float(fps), (640, 480), is_color)
    for image in images:
        #print(image)
        #if not os.path.exists(image):
        #    raise FileNotFoundError(image)
        #img = cv.imread(image)
        #img = cv.resize(img, (640, 480))
        vid.write(image)
    vid.release()
    return vid

#################################
########## RetNetPose ###########
#################################
def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
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


def load_model(model_path):

    check_keras_version()

    #if args.gpu:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    anchor_params = None
    backbone = 'resnet50'

    print('Loading model, this may take a second...')
    model = models.load_model(model_path, backbone_name=backbone)
    graph = tf.get_default_graph()
    model = models.convert_model(model, anchor_params=anchor_params) # convert model

    # print model summary
    print(model.summary())

    return model, graph


def run_estimation(image, model, score_threshold, graph):
    poses = []
    boxes2comp = []

    #cv2.imwrite('/home/sthalham/retnetpose_image.jpg', image)

    #if np.nanmax(image_dep) < 1000.0: # orbbec
    #    image_dep * 1000.0

    #image_dep[image_dep > 2000.0] = 0
    #scaCro = 255.0 / np.nanmax(image_dep)
    #cross = np.multiply(image_dep, scaCro)
    #image = cross.astype(np.uint8)
    #image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)

    #cv2.imwrite('/home/sthalham/retnetpose_image.jpg', image)

    if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

    with graph.as_default():
        boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct boxes for image scale
    #boxes /= scale   # may be relevant at some point

    print(scores[:, :8])
    print(labels[:, :8])

    # change to (x, y, w, h) (MS COCO standard)
    boxes[:, :, 2] -= boxes[:, :, 0]
    boxes[:, :, 3] -= boxes[:, :, 1]

    det1 = False
    det2 = False
    det3 = False
    det4 = False
    det5 = False
    det6 = False
    det7 = False
    det8 = False

    #print('new image')

    # compute predicted labels and scores
    for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
        # scores are sorted, so we can break

        if score < score_threshold:
            continue

        # ugly workaround for IoU exception
        ov_detect = False
        for bb in boxes2comp:
            ovlap = boxoverlap(box, bb)
            if ovlap > 0.5:
                ov_detect = True

        boxes2comp.append(box)
        if ov_detect is True:
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
        elif label == 4 and det5 == False:
            det5 = True
        elif label == 5 and det6 == False:
            det6 = True
        elif label == 6 and det7 == False:
            det7 = True
        elif label == 7 and det8 == False:
            det8 = True
        else:
            continue

        control_points = box3D[(label), :]

        dC = label+1

        obj_points = np.ascontiguousarray(threeD_boxes[dC - 1, :, :], dtype=np.float32)
        est_points = np.ascontiguousarray(np.asarray(control_points, dtype=np.float32).T, dtype=np.float32).reshape(
            (8, 1, 2))

        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                           imagePoints=est_points, cameraMatrix=K,
                                                           distCoeffs=None, rvec=None, tvec=None,
                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                           reprojectionError=8.0, confidence=0.99,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(orvec)

        pcd_img = create_point_cloud(image_dep, fxkin, fykin, cxkin, cykin, 0.001)
        pcd_img = pcd_img.reshape((480, 640, 3))[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2]), :]
        pcd_img = pcd_img.reshape((pcd_img.shape[0] * pcd_img.shape[1], 3))
        pcd_crop = open3d.PointCloud()
        pcd_crop.points = open3d.Vector3dVector(pcd_img)
        pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
        # open3d.draw_geometries([pcd_crop, pcd_model])

        guess = np.zeros((4, 4), dtype=np.float32)
        guess[:3, :3] = rmat
        guess[:3, 3] = otvec.T
        guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

        if dC == 1:
            pcd_model = pcd_model_1
        elif dC == 2:
            pcd_model = pcd_model_2
        elif dC == 3:
            pcd_model = pcd_model_3
        elif dC == 4:
            pcd_model = pcd_model_4
        elif dC == 5:
            pcd_model = pcd_model_5
        elif dC == 6:
            pcd_model = pcd_model_6
        elif dC == 7:
            pcd_model = pcd_model_7
        elif dC == 8:
            pcd_model = pcd_model_8
        reg_p2p = open3d.registration_icp(pcd_model, pcd_crop, 0.015, guess,
                                          open3d.TransformationEstimationPointToPoint())
        R_est = reg_p2p.transformation[:3, :3]
        t_est = reg_p2p.transformation[:3, 3]
        rot = tf3d.quaternions.mat2quat(R_est)

        pose = [dC, score, t_est[0], t_est[1], t_est[2], rot[0], rot[1], rot[2], rot[3]]
        poses.append(pose)

        font = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (int(box[0]) + 5, int(box[1]) + int(box[3]) - 5)
        fontScale = 0.5
        fontColor = (25, 215, 250)
        fontthickness = 2
        lineType = 2
        gtText = objectNames[dC-1]
        print(gtText)

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
        axisPoints, _ = cv2.projectPoints(points, R_est, t_est*1000.0, K, (0, 0, 0, 0))

        image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
        image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
        image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

    scaCro = 255.0 / np.nanmax(image)
    visImg = np.multiply(image, scaCro)
    visImg = visImg.astype(np.uint8)
    #cv2.imwrite('/home/sthalham/retnetpose_detects.jpg', visImg)
    #cv2.imwrite('/home/mmassist/mmassist/Detection_TUWien/retnetpose_detects.jpg', visImg)

    return image


if __name__ == '__main__':

    score_threshold = 0.5
    icp_threshold = 0.15
    model_path = '/home/sthalham/data/MMAssist_Fronius/depth_weights/fronius_M24_new_44.h5'

    model, graph = load_model(model_path)

    images = []
    vidcap = cv2.VideoCapture('/home/sthalham/Downloads/Video3_1_Depth.avi')

    count = 0
    success = True
    while vidcap.isOpened():
        #cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('Read a new frame: ', success)
        count += 1
        print(count)
        #if(count > 1000):
        #    break

        det_poses = run_estimation(image, model, score_threshold, graph)
        images.append(det_poses)

    out = '/home/sthalham/fronius3_1.avi'
    make_video(images, out)
    print("Job done! clip exported to: ", out)