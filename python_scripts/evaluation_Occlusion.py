#!/usr/bin/env python

import sys
import os
import yaml
import cv2
import numpy as np
import json
import transforms3d as tf3d
import math
import copy
import open3d
from scipy import spatial
from pose_error import vsd, reproj, add, adi, re, te
import ply_loader

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

threeD_boxes = np.ndarray((15, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[0.038, 0.039, 0.046],  # ape [76, 78, 92]
                                     [0.038, 0.039, -0.046],
                                     [0.038, -0.039, -0.046],
                                     [0.038, -0.039, 0.046],
                                     [-0.038, 0.039, 0.046],
                                     [-0.038, 0.039, -0.046],
                                     [-0.038, -0.039, -0.046],
                                     [-0.038, -0.039, 0.046]])
threeD_boxes[1, :, :] = np.array([[0.108, 0.061, 0.1095],  # benchvise [216, 122, 219]
                                     [0.108, 0.061, -0.1095],
                                     [0.108, -0.061, -0.1095],
                                     [0.108, -0.061, 0.1095],
                                     [-0.108, 0.061, 0.1095],
                                     [-0.108, 0.061, -0.1095],
                                     [-0.108, -0.061, -0.1095],
                                     [-0.108, -0.061, 0.1095]])
threeD_boxes[2, :, :] = np.array([[0.083, 0.0825, 0.037],  # bowl [166, 165, 74]
                                     [0.083, 0.0825, -0.037],
                                     [0.083, -0.0825, -0.037],
                                     [0.083, -0.0825, 0.037],
                                     [-0.083, 0.0825, 0.037],
                                     [-0.083, 0.0825, -0.037],
                                     [-0.083, -0.0825, -0.037],
                                     [-0.083, -0.0825, 0.037]])
threeD_boxes[3, :, :] = np.array([[0.0685, 0.0715, 0.05],  # camera [137, 143, 100]
                                     [0.0685, 0.0715, -0.05],
                                     [0.0685, -0.0715, -0.05],
                                     [0.0685, -0.0715, 0.05],
                                     [-0.0685, 0.0715, 0.05],
                                     [-0.0685, 0.0715, -0.05],
                                     [-0.0685, -0.0715, -0.05],
                                     [-0.0685, -0.0715, 0.05]])
threeD_boxes[4, :, :] = np.array([[0.0505, 0.091, 0.097],  # can [101, 182, 194]
                                     [0.0505, 0.091, -0.097],
                                     [0.0505, -0.091, -0.097],
                                     [0.0505, -0.091, 0.097],
                                     [-0.0505, 0.091, 0.097],
                                     [-0.0505, 0.091, -0.097],
                                     [-0.0505, -0.091, -0.097],
                                     [-0.0505, -0.091, 0.097]])
threeD_boxes[5, :, :] = np.array([[0.0335, 0.064, 0.0585],  # cat [67, 128, 117]
                                     [0.0335, 0.064, -0.0585],
                                     [0.0335, -0.064, -0.0585],
                                     [0.0335, -0.064, 0.0585],
                                     [-0.0335, 0.064, 0.0585],
                                     [-0.0335, 0.064, -0.0585],
                                     [-0.0335, -0.064, -0.0585],
                                     [-0.0335, -0.064, 0.0585]])
threeD_boxes[6, :, :] = np.array([[0.059, 0.046, 0.0475],  # mug [118, 92, 95]
                                     [0.059, 0.046, -0.0475],
                                     [0.059, -0.046, -0.0475],
                                     [0.059, -0.046, 0.0475],
                                     [-0.059, 0.046, 0.0475],
                                     [-0.059, 0.046, -0.0475],
                                     [-0.059, -0.046, -0.0475],
                                     [-0.059, -0.046, 0.0475]])
threeD_boxes[7, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[8, :, :] = np.array([[0.052, 0.0385, 0.043],  # duck [104, 77, 86]
                                     [0.052, 0.0385, -0.043],
                                     [0.052, -0.0385, -0.043],
                                     [0.052, -0.0385, 0.043],
                                     [-0.052, 0.0385, 0.043],
                                     [-0.052, 0.0385, -0.043],
                                     [-0.052, -0.0385, -0.043],
                                     [-0.052, -0.0385, 0.043]])
threeD_boxes[9, :, :] = np.array([[0.075, 0.0535, 0.0345],  # eggbox [150, 107, 69]
                                     [0.075, 0.0535, -0.0345],
                                     [0.075, -0.0535, -0.0345],
                                     [0.075, -0.0535, 0.0345],
                                     [-0.075, 0.0535, 0.0345],
                                     [-0.075, 0.0535, -0.0345],
                                     [-0.075, -0.0535, -0.0345],
                                     [-0.075, -0.0535, 0.0345]])
threeD_boxes[10, :, :] = np.array([[0.0185, 0.039, 0.0865],  # glue [37, 78, 173]
                                     [0.0185, 0.039, -0.0865],
                                     [0.0185, -0.039, -0.0865],
                                     [0.0185, -0.039, 0.0865],
                                     [-0.0185, 0.039, 0.0865],
                                     [-0.0185, 0.039, -0.0865],
                                     [-0.0185, -0.039, -0.0865],
                                     [-0.0185, -0.039, 0.0865]])
threeD_boxes[11, :, :] = np.array([[0.0505, 0.054, 0.04505],  # holepuncher [101, 108, 91]
                                     [0.0505, 0.054, -0.04505],
                                     [0.0505, -0.054, -0.04505],
                                     [0.0505, -0.054, 0.04505],
                                     [-0.0505, 0.054, 0.04505],
                                     [-0.0505, 0.054, -0.04505],
                                     [-0.0505, -0.054, -0.04505],
                                     [-0.0505, -0.054, 0.04505]])
threeD_boxes[12, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[13, :, :] = np.array([[0.129, 0.059, 0.0705],  # iron [258, 118, 141]
                                     [0.129, 0.059, -0.0705],
                                     [0.129, -0.059, -0.0705],
                                     [0.129, -0.059, 0.0705],
                                     [-0.129, 0.059, 0.0705],
                                     [-0.129, 0.059, -0.0705],
                                     [-0.129, -0.059, -0.0705],
                                     [-0.129, -0.059, 0.0705]])
threeD_boxes[14, :, :] = np.array([[0.047, 0.0735, 0.0925],  # phone [94, 147, 185]
                                     [0.047, 0.0735, -0.0925],
                                     [0.047, -0.0735, -0.0925],
                                     [0.047, -0.0735, 0.0925],
                                     [-0.047, 0.0735, 0.0925],
                                     [-0.047, 0.0735, -0.0925],
                                     [-0.047, -0.0735, -0.0925],
                                     [-0.047, -0.0735, 0.0925]])

model_radii = np.array([0.041, 0.0928, 0.0675, 0.0633, 0.0795, 0.052, 0.0508, 0.0853, 0.0445, 0.0543, 0.048, 0.05, 0.0862, 0.0888, 0.071])


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


def draw_axis(img, poses):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    rot = np.zeros((3,), dtype=np.float32)

    #rot = tf3d.euler.quat2euler(poses[3:7])
    rotMat = tf3d.quaternions.quat2mat(poses[3:7])
    rot, _ = cv2.Rodrigues(rotMat)
    tra = poses[0:3] * 1000.0
    K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3,3)
    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))

    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


def listDiff(first, second):
    # second = set(second)
    return [item for item in first if item not in second]


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


if __name__ == "__main__":

    #root = '/home/sthalham/data/T-less_Detectron/output/linemodArtiHAA07062018/test/coco_2014_val/generalized_rcnn/'  # path to train samples, depth + rgb
    #root = "/home/sthalham/data/results/"
    #root = "/home/sthalham/workspace/creation-pipeline/json_results/"
    #jsons = root + 'full_1214.json'
    #jsons = root + 'results_lm_baseline_X_101_B64.json'
    #root = '/home/sthalham/data/prepro/linemod_real/out/test/coco_2014_val/generalized_rcnn/'
    #jsons = root + 'bbox_coco_2014_val_results.json'
    root = '/home/sthalham/workspace/RetNetPose/'
    #jsons = root + '3Dbox_linemod.json'
    #root = '/home/sthalham/workspace/MMAssist_pipeline/RetinaNet/'
    #jsons = root + 'val_61.json'
    jsons = root + 'val_bbox_results.json'
    #jsons = root + 'rgd20_results.json'
    model_path = "/home/sthalham/data/LINEMOD/models/"

    visu = True

    json_data = open(jsons).read()
    data = json.loads(json_data)

    testData = '/home/sthalham/data/LINEMOD/test/'
    sub = os.listdir(testData)

    absObjs = 0

    gtCatLst = [0] * 16
    detCatLst = [0] * 16
    falsePosLst = [0] * 16
    falseNegLst = [0] * 16

    detFrac = [False] * 16
    allImg = 18273

    xyzD = []
    less5cm = []
    rotD = []
    less5deg = []
    less5 = []

    rep_e = []
    rep_less5 = []

    add_e = []
    add_less_d = []

    vsd_e = []
    vsd_less_t = []

    proImg = 0

    for s in sub:
        print(s)

        #if s == '15' or s == '13' or s == '06' or s == '01' or s == '09' or s == '02' or s == '07' or s == '04' or s == '12' or s == '08' or s == '11':
        if s != '02':
            continue

        rgbPath = testData + s + "/rgb/"
        depPath = testData + s + "/depth/"
        gtPath = testData + s + "/gt.yml"

        # load mesh and transform
        ply_path = model_path + 'obj_' + s + '.ply'
        #pcd_model = open3d.read_point_cloud(ply_path)
        #pcd_model.paint_uniform_color(np.array([0.0, 0.0, 0.99]))
        model_vsd = ply_loader.load_ply(ply_path)
        pcd_model = open3d.PointCloud()
        pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        #open3d.draw_geometries([pcd_model])
        model_vsd_mm = copy.deepcopy(model_vsd)
        model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0

        with open(gtPath, 'r') as streamGT:
            gtYML = yaml.load(streamGT)

        subsub = os.listdir(rgbPath)

        counter = 0

        posAng = 0  
        fn = 0
        for ss in subsub:

            posevis = []
            #if ss != '0033.png':
            #    continue
            #print(ss)

            proImg = proImg + 1
            #if proImg % 100 == 0:
            #    print('----------------------------------------------------------------------------')
            #    print('------------Processing image ', proImg, ' / ', allImg, ' -------------------')
            #    print('----------------------------------------------------------------------------')

            imgname = ss

            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss
            # print('processing image: ', rgbImgPath)

            if ss.startswith('000'):
                ss = ss[3:]
            elif ss.startswith('00'):
                ss = ss[2:]
            elif ss.startswith('0'):
                ss = ss[1:]
            ss = ss[:-4]

            template = '00000'
            s = int(s)
            ssm = int(ss) + 1
            pre = (s - 1) * 1296
            img_id = pre + ssm
            tempSS = template[:-len(str(img_id))]

            imgNum = str(img_id)
            imgNam = tempSS + imgNum + '.jpg'
            iname = str(imgNam)

            gtImg = gtYML[int(ss)]
            gtBoxes = []
            gtCats = []
            gtPoses = []
            gtRots = []
            for gt in gtImg:
                #if gt['obj_id'] == s:
                gtBoxes.append(gt['obj_bb'])
                gtCats.append(gt['obj_id'])
                gtPoses.append(gt['cam_t_m2c'])
                gtRots.append(gt["cam_R_m2c"])

            detBoxes = []
            detCats = []
            detSco = []
            detPoses = []
            detSeg = []
            detIMG = []
            for det in data:
                if det['image_id'] == img_id:
                    detIMG = det['image_id']
                    detBoxes.append(det['bbox'])
                    detCats.append(det['category_id'])
                    detSco.append(det['score'])
                    detPoses.append(det['pose'])

            img = cv2.imread(depImgPath, -1)
            img_gt = cv2.imread(rgbImgPath, -1)

            if len(detBoxes) < 1 and detIMG == img_id:
                #for i in gtCats:
                #    gtCatLst[i] = gtCatLst[i] + 1
                gtCatLst[s] += 1
                fn += 1

            #else:
            elif detIMG:

                for i, dC in enumerate(detCats):

                    obj_points = np.ascontiguousarray(threeD_boxes[dC - 1, :, :], dtype=np.float32)
                    est_points = np.ascontiguousarray(np.asarray(detPoses[i], dtype=np.float32).T,
                                                      dtype=np.float32).reshape(
                        (8, 1, 2))

                    K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
                    retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                       imagePoints=est_points, cameraMatrix=K,
                                                                       distCoeffs=None, rvec=None, tvec=None,
                                                                       useExtrinsicGuess=False, iterationsCount=100,
                                                                       reprojectionError=8.0, confidence=0.99,
                                                                       flags=cv2.SOLVEPNP_ITERATIVE)
                    rmat, _ = cv2.Rodrigues(orvec)
                    #rot = tf3d.quaternions.mat2quat(R_est)
                    pose = np.concatenate((np.array(otvec[:, 0].T, dtype=np.float32), np.array(rmat, dtype=np.float32).reshape(9)),
                                          axis=0)
                    posevis.append(pose.tolist())

                posevis = [x for _, x in sorted(zip(detCats, posevis))]
                detBoxes = [x for _, x in sorted(zip(detCats, detBoxes))]
                detCats.sort()
                print(detCats)
                if visu == True:

                    img = copy.deepcopy(img_gt)


                    '''
                    for i, cat in enumerate(detCats):

                        if detCats[i] == 3 or detCats[i] == 7:
                            continue
                        print(detCats[i])

                        bb = detBoxes[i]

                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (int(bb[0])+5, int(bb[1])+int(bb[3])-5)
                        fontScale = 0.5
                        fontColor = (25, 215, 250)
                        fontthickness = 2
                        lineType = 2


                        if detCats[i] == 1: cate = 'Ape'
                        elif detCats[i] == 2: cate = 'Benchvise'
                        elif detCats[i] == 3: cate = 'Bowl'
                        elif detCats[i] == 4: cate = 'Camera'
                        elif detCats[i] == 5: cate = 'Can'
                        elif detCats[i] == 6: cate = 'Cat'
                        elif detCats[i] == 7: cate = 'Cup'
                        elif detCats[i] == 8: cate = 'Driller'
                        elif detCats[i] == 9: cate = 'Duck'
                        elif detCats[i] == 10: cate = 'Eggbox'
                        elif detCats[i] == 11: cate = 'Glue'
                        elif detCats[i] == 12: cate = 'Holepuncher'
                        elif detCats[i] == 13: cate = 'Iron'
                        elif detCats[i] == 14: cate = 'Lamp'
                        elif detCats[i] == 15: cate = 'Phone'
                        gtText = cate
                        #gtText = cate + " / " + str(detSco[i])

                        fontColor2 = (0, 0, 0)
                        fontthickness2 = 4
                        cv2.putText(img, gtText,
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor2,
                                        fontthickness2,
                                        lineType)

                        cv2.putText(img, gtText,
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        fontthickness,
                                        lineType)
                    '''

                    gtRots = [x for _, x in sorted(zip(gtCats, gtRots))]
                    gtPoses = [x for _, x in sorted(zip(gtCats, gtPoses))]
                    gtBoxes = [x for _, x in sorted(zip(gtCats, gtBoxes))]
                    gtCats.sort()
                    for i, cat in enumerate(gtCats):

                        if cat == 2:
                            continue

                        '''
                        bb = gtBoxes[i]

                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (10, 30)
                        fontScale = 1
                        lineType = 2

                        gtText = 'Ground Truth'

                        fontColor2 = (255, 255, 255)
                        fontthickness2 = 5
                        cv2.putText(img_gt, gtText,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor2,
                                    fontthickness2,
                                    lineType)

                        fontthickness = 2
                        fontColor = (0, 0, 0)
                        cv2.putText(img_gt, gtText,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    fontthickness,
                                    lineType)
                        '''

                        rot = gtRots[i]
                        tra = gtPoses[i]

                        rot = np.asarray(rot, dtype=np.float32).reshape((3, 3))
                        tra = np.asarray(tra, dtype=np.float32) * 0.001
                        tDbox = rot.dot(threeD_boxes[cat - 1, :, :].T).T
                        tDbox = tDbox + np.repeat(tra[:, np.newaxis], 8, axis=1).T

                        box3D = toPix_array(tDbox)
                        box3D = np.reshape(box3D, (16))
                        pose = box3D

                        colR = 0
                        colG = 255
                        colB = 0

                        colR1 = 0
                        colG1 = 255
                        colB1 = 0

                        colR2 = 0
                        colG2 = 255
                        colB2 = 0

                        colR3 = 0
                        colG3 = 255
                        colB3 = 0

                        colR4 = 0
                        colG4 = 255
                        colB4 = 0

                        colR5 = 0
                        colG5 = 255
                        colB5 = 0

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (colR, colG, colB), 4)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (colR, colG, colB), 4)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (colR1, colG1, colB1), 4)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (colR1, colG1, colB1), 4)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR2, colG2, colB2), 4)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR2, colG2, colB2), 4)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR5, colG5, colB5), 4)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR5, colG5, colB5), 4)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), (colR3, colG3, colB3),
                                       4)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), (colR3, colG3, colB3),
                                       4)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), (colR4, colG4, colB4),
                                       4)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), (colR4, colG4, colB4),
                                       4)

                    for i, Dbox in enumerate(posevis):

                        if detCats[i] == 3 or detCats[i] == 7:
                            continue
                        print(detCats[i])

                        rot = Dbox[3:]
                        tra = Dbox[:3]
                        # pose = np.concatenate([tra[:, 0].transpose(), rot.transpose()])

                        rot = np.asarray(rot, dtype=np.float32).reshape((3, 3))
                        tra = np.asarray(tra, dtype=np.float32)
                        '''
                        if detCats[i] == 1:
                            cate = '01'
                            col = [0.99, 0.0, 0.00]
                        elif detCats[i] == 2:
                            cate = '02'
                            col = [0.00, 0.99, 0.00]
                        elif detCats[i] == 5:
                            cate = '05'
                            col = [0.00, 0.00, 0.99]
                        elif detCats[i] == 6:
                            cate = '06'
                            col = [0.99, 0.00, 0.5]
                        elif detCats[i] == 8:
                            cate = '08'
                            col = [0.00, 0.99, 0.5]
                        elif detCats[i] == 9:
                            cate = '09'
                            col = [0.99, 0.5, 0.00]
                        elif detCats[i] == 10:
                            cate = '10'
                            col = [0.5, 0.00, 0.99]
                        elif detCats[i] == 11:
                            cate = '11'
                            col = [0.00, 0.5, 0.99]
                        elif detCats[i] == 12:
                            cate = '12'
                            col = [0.5, 0.99, 0.0]

                        ply_path = model_path + 'obj_' + cate + '.ply'
                        model_vsd = ply_loader.load_ply(ply_path)
                        pcd_model = open3d.PointCloud()
                        pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
                        pcd_model.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                        open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30))
                        # open3d.draw_geometries([pcd_model])
                        model_vsd_mm = copy.deepcopy(model_vsd)
                        model_vsd_mm['pts'] = model_vsd_mm['pts']

                        tDbox = rot.dot(model_vsd_mm['pts'].T).T
                        tDbox = tDbox + np.repeat(tra[:, np.newaxis], tDbox.shape[0], axis=1).T

                        box3D = toPix_array(tDbox)

                        print(col)
                        npcol = np.array(col, dtype=np.uint8) *255
                        for j in range(0, box3D.shape[0]):
                            print(box3D[j, 0])
                            print(box3D[j, 1])
                            print(img.shape)
                            print(npcol)
                            img[int(box3D[j,0]), int(box3D[j,1]), :] = npcol

                        cv2.imwrite('/home/sthalham/visTests/proj_test.jpg', img)

                        bb = detBoxes[i]

                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (10, 30)
                        fontScale = 1
                        lineType = 2

                        gtText = 'Estimates'

                        fontColor2 = (255, 255, 255)
                        fontthickness2 = 5
                        cv2.putText(img, gtText,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor2,
                                    fontthickness2,
                                    lineType)

                        fontthickness = 2
                        fontColor = (0, 0, 0)
                        cv2.putText(img, gtText,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    fontthickness,
                                    lineType)
                        '''

                        tDbox = rot.dot(threeD_boxes[detCats[i] - 1, :, :].T).T
                        tDbox = tDbox + np.repeat(tra[:, np.newaxis], 8, axis=1).T

                        box3D = toPix_array(tDbox)
                        box3D = np.reshape(box3D, (16))
                        pose = box3D

                        colR = 255
                        colG = 0
                        colB = 0

                        colR1 = 255
                        colG1 = 0
                        colB1 = 0

                        colR2 = 255
                        colG2 = 0
                        colB2 = 0

                        colR3 = 255
                        colG3 = 0
                        colB3 = 0

                        colR4 = 255
                        colG4 = 0
                        colB4 = 0

                        colR5 = 255
                        colG5 = 0
                        colB5 = 0

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (colR, colG, colB), 4)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (colR, colG, colB), 4)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (colR1, colG1, colB1),
                                       4)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (colR1, colG1, colB1),
                                       4)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR2, colG2, colB2),
                                       4)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR2, colG2, colB2),
                                       4)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR5, colG5, colB5),
                                       4)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR5, colG5, colB5),
                                       4)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                                       (colR3, colG3, colB3),
                                       4)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                                       (colR3, colG3, colB3),
                                       4)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                                       (colR4, colG4, colB4),
                                       4)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                                       (colR4, colG4, colB4),
                                       4)

                    for i, cat in enumerate(gtCats):
                        if cat == 2:
                            continue

                        bb = gtBoxes[i]

                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (int(bb[0]) + 5, int(bb[1]) + int(bb[3]) - 5)
                        fontScale = 0.5
                        fontColor = (0, 0, 255)
                        fontthickness = 2
                        lineType = 2

                        if cat == 1: cate = 'Ape'
                        elif cat == 2: cate = 'Benchvise'
                        elif cat == 3: cate = 'Bowl'
                        elif cat == 4: cate = 'Camera'
                        elif cat == 5: cate = 'Can'
                        elif cat == 6: cate = 'Cat'
                        elif cat == 7: cate = 'Cup'
                        elif cat == 8: cate = 'Driller'
                        elif cat == 9: cate = 'Duck'
                        elif cat == 10: cate = 'Eggbox'
                        elif cat == 11: cate = 'Glue'
                        elif cat == 12: cate = 'Holepuncher'
                        elif cat == 13: cate = 'Iron'
                        elif cat == 14: cate = 'Lamp'
                        elif cat == 15: cate = 'Phone'
                        gtText = cate
                        #gtText = cate + " / " + str(detSco[i])

                        fontColor2 = (0, 0, 0)
                        fontthickness2 = 4
                        cv2.putText(img_gt, gtText,
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor2,
                                        fontthickness2,
                                        lineType)

                        cv2.putText(img_gt, gtText,
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        fontthickness,
                                        lineType)

                    #img_vid = np.ones([600, 800, 3], dtype=np.uint8)
                    #img_vid[21:501, 81:721, :]= img
                    img_vid = img

                    font = cv2.FONT_HERSHEY_COMPLEX
                    bottomLeftCornerOfText = (20, 535)
                    fontScale = 1
                    lineType = 2

                    gtText = 'Ground Truth'

                    fontColor2 = (0, 255, 0)
                    fontthickness2 = 3
                    cv2.putText(img_vid, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor2,
                                fontthickness2,
                                lineType)

                    font = cv2.FONT_HERSHEY_COMPLEX
                    bottomLeftCornerOfText = (615, 535)
                    fontScale = 1
                    lineType = 2

                    gtText = 'Estimates'

                    fontColor2 = (255, 0, 0)
                    fontthickness2 = 3
                    cv2.putText(img_vid, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor2,
                                fontthickness2,
                                lineType)

                    name_est = '/home/sthalham/Documents/3DV_2019/imgs4paper/' + rgbImgPath[-8:-4] + '.jpg'
                    print(name_est)
                    cv2.imwrite(name_est, img_vid)

                    #scaCro = 255.0 / np.nanmax(image_dep)
                    #visImg = np.multiply(image_dep, scaCro)
                    #visImg = visImg.astype(np.uint8)
                    #name_est = '/home/sthalham/visTests/img_dep.jpg'
                    #cv2.imwrite(name_est, visImg)

                    print('STOP')

            else:
                pass
            #falseNegLst[s] = fn

        #detAcc = detCatLst[s] / gtCatLst[s]
        #print('accuracy category ', s, ': ', detAcc)

        #falseNegLst[s] = fn
        #if (detCatLst[s] + falsePosLst[s]) == 0:
        #    detPre = 0.0
        #else:
        #    detPre = detCatLst[s] / (detCatLst[s] + falsePosLst[s])
        #if (detCatLst[s] + falseNegLst[s]) == 0:
        #    detRec = 0.0
        #else:
        #    detRec = detCatLst[s] / (detCatLst[s] + falseNegLst[s])

        #print('precision category ', s, ': ', detPre)
        #print('recall category ', s, ': ', detRec)
        #if angDif[s] == 0 or couPos[s] == 0:
        #    angDif[s] = 0.0
        #else:
        #    angDif[s] = angDif[s] / couPos[s]
        #    print(s, ' (amount poses): ', couPos[s])

        #print('STOP')


    #dataset_xyz_diff = (sum(xyzD) / len(xyzD))
    #dataset_depth_diff = (sum(zD) / len(zD))
    #less5cm = len(less5cm)/len(xyzD)
    #less10cm = len(less10cm) / len(xyzD)
    #less15cm = len(less15cm) / len(xyzD)
    #less20cm = len(less20cm) / len(xyzD)
    #less25cm = len(less25cm) / len(xyzD)
    #less5deg = len(less5deg) / len(rotD)
    #less10deg = len(less10deg) / len(rotD)
    #less15deg = len(less15deg) / len(rotD)
    #less20deg = len(less20deg) / len(rotD)
    #less25deg = len(less25deg) / len(rotD)
    less_55 = len(less5) / len(rotD) * 100.0
    less_repr_5 = len(rep_less5) / len(rep_e) * 100.0
    less_add_d = len(add_less_d) / len(add_e) * 100.0
    less_vsd_t = len(vsd_less_t) / len(vsd_e) * 100.0

    #print('dataset recall: ', dataset_recall, '%')
    #print('dataset precision: ', dataset_precision, '%')
    #print('linemod::percent below 5 cm: ', less5cm, '%')
    #print('linemod::percent below 10 cm: ', less10cm, '%')
    #print('linemod::percent below 15 cm: ', less15cm, '%')
    #print('linemod::percent below 20 cm: ', less20cm, '%')
    #print('linemod::percent below 25 cm: ', less25cm, '%')
    #print('linemod::percent below 5 deg: ', less5deg, '%')
    #print('linemod::percent below 10 deg: ', less10deg, '%')
    #print('linemod::percent below 15 deg: ', less15deg, '%')
    #print('linemod::percent below 20 deg: ', less20deg, '%')
    #print('linemod::percent below 25 deg: ', less25deg, '%')

    print('linemod::percent poses below 5cm and 5°: ', less_55, '%')
    print('linemod::percent VSD below tau 0.02m: ',  less_vsd_t, '%')
    print('linemod::percent reprojection below 5 pixel: ', less_repr_5, '%')
    print('linemod::percent ADD below model diameter: ', less_add_d, '%')


    # Precision = True positive / (True positive + False positive)
    # Recall = True positive / (True positive + False negative)

    detPre = [0] * 16
    detRec = [0] * 16
    detRot = [0] * 16
    detless5 = [0] * 16

    np.set_printoptions(precision=2)
    for ind, cat in enumerate(gtCatLst):
        #if ind == 0 or ind == 2 or ind == 3 or ind == 4 or ind == 7 or ind == 13 or ind == 14 or ind == 15:
        if ind == 0:
            continue

        if (detCatLst[ind] + falsePosLst[ind]) == 0:
            detPre[ind] = 0.0
        else:
            detPre[ind] = detCatLst[ind] / (detCatLst[ind] + falsePosLst[ind])
        if (detCatLst[ind] + falseNegLst[ind]) == 0:
            detRec[ind] = 0.0
        else:
            detRec[ind] = detCatLst[ind] / (detCatLst[ind] + falseNegLst[ind])
            #detRec[ind] = detCatLst[ind] / (gtCatLst[ind])
        if (angDif[ind] ) == 0:
            detRot[ind] = 0.0
            detless5[ind] = 0
        else:
            detRot[ind] = angDif[ind]
            detless5[ind] = less5[ind] /couPos[ind]

        print('precision category ', ind, ': ', detPre[ind])
        print('recall category ', ind, ': ', detRec[ind], (detCatLst[ind] + falseNegLst[ind]))
        #print('pose Acc ', ind, ': ', detRot[ind]/couPos[ind])
        print('< 5° ', ind, ': ', detless5[ind])

    #print(detCatLst, ' sum: ', sum(detCatLst))
    #print(gtCatLst, ' sum: ', sum(gtCatLst))

    #print('accuracy overall: ', sum(detAcc)/len(detAcc))
    print('mP: ', sum(detPre) / (len(detPre)))
    print('mR: ', sum(detRec) / len(detRec))
    #print('AP: ', sum(detPre) / 8)
    #print('AR: ', sum(detRec) / 8)
    #print('mdRot: ', sum(posDif) / len(posDif))
    #print('mdRot: ', sum(depDif) / len(depDif))
    #print('mdRot: ', sum(angDif) / sum(couPos))

    print('mean < 5: ', sum(less5) / sum(couPos))
    print('mdRot: ', sum(detRot) / sum(couPos))




