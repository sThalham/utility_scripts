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
    jsons = root + 'val_bbox_results.json'
    #jsons = root + 'rgd20_results.json'
    model_path = "/home/sthalham/data/LINEMOD/models/"

    visu = False

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
        if s != '14':
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
            if proImg % 100 == 0:
                print('----------------------------------------------------------------------------')
                print('------------Processing image ', proImg, ' / ', allImg, ' -------------------')
                print('----------------------------------------------------------------------------')

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
                if gt['obj_id'] == s:
                    gtBoxes.append(gt['obj_bb'])
                    gtCats.append(gt['obj_id'])
                    gtPoses.append(gt['cam_t_m2c'])
                    gtRots.append(gt["cam_R_m2c"])

            '''
            img = cv2.imread(depImgPath, -1)
            img_gt = cv2.imread(rgbImgPath, -1)
            for i, bb in enumerate(gtBoxes):

                if gtCats[i] == 12:
                    img_crop = img[bb[1]:(bb[1]+bb[3]), bb[0]:(bb[0]+bb[2])]
                    R = np.asarray(gtRots[i], dtype=np.float32)
                    rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                    print(bb)
                    print(rot)
                    print(str(gtPoses[i]), (np.asarray(tf3d.euler.mat2euler(R.reshape(3,3))) * (180.0/math.pi)))
                    print(str(gtPoses[i]), rot)
                    name = '/home/sthalham/visTests/crop.png'
                    cv2.imwrite(name, img_crop)
            '''

                #rot = tf3d.quaternions.mat2quat(gtRotIt)
                #tra = np.asarray(gtPoses[i], dtype=np.float32)
                #pose = np.concatenate([tra.transpose(), rot.transpose()])
                #draw_axis(img_gt, pose)

            #name = '/home/sthalham/visTests/gt.png'
            #cv2.imwrite(name, img_gt)

            absObjs = absObjs + len(gtCats)  # increment all

            detBoxes = []
            detCats = []
            detSco = []
            detPoses = []
            detSeg = []
            detIMG = []
            for det in data:
                if det['image_id'] == img_id:
                    detIMG = det['image_id']
                    if det['category_id'] == s:
                        #if det['score'] > 0.55:
                        detBoxes.append(det['bbox'])
                        detCats.append(det['category_id'])
                        detSco.append(det['score'])
                        detPoses.append(det['pose'])
                        #detSeg.append(det['segmentation'])
                        #print(det['pose'])

            #temp = False
            #if temp == True:
            if len(detBoxes) < 1 and detIMG == img_id:
                #for i in gtCats:
                #    gtCatLst[i] = gtCatLst[i] + 1
                gtCatLst[s] += 1
                fn += 1

            #else:
            elif detIMG:
                detBoxes = [detBoxes[detCats.index(s)]]
                detSco = [detSco[detCats.index(s)]]
                detCats = [detCats[detCats.index(s)]]
                detPoses = [detPoses[detCats.index(s)]]
                #detFrac[s] = True

                falsePos = []
                truePos = []
                #truePos = 0
                fnCur = 1
                for i, dC in enumerate(detCats):
                    for j, gC in enumerate(gtCats):
                        if dC is gC:

                            b1 = np.array([detBoxes[i][0], detBoxes[i][1], detBoxes[i][0] + detBoxes[i][2], detBoxes[i][1] + detBoxes[i][3]])
                            b2 = np.array([gtBoxes[j][0], gtBoxes[j][1], gtBoxes[j][0] + gtBoxes[j][2], gtBoxes[j][1] + gtBoxes[j][3]])
                            IoU = boxoverlap(b1, b2)
                            # occurences of 2 or more instances not possible in LINEMOD
                            if IoU > 0.5:
                                truePos.append(dC)
                                fnCur = 0

                                print('--------------------- BBox center as initial estimate -------------------')
                                image_dep = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
                                dep_val = image_dep[int(b1[1] + (detBoxes[i][3] * 0.5)), int(b1[0] + (detBoxes[i][2] * 0.5))] * 0.001
                                dep = dep_val + model_radii[dC-1]

                                x_o = (((b1[0] + (detBoxes[i][2] * 0.5)) - cxkin) * dep) / fxkin
                                y_o = (((b1[1] + (detBoxes[i][3] * 0.5)) - cykin) * dep) / fykin

                                irvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                                itvec = np.array([x_o, y_o, dep], dtype=np.float32)

                                print('--------------------- PnP Pose Estimation -------------------')
                                obj_points = np.ascontiguousarray(threeD_boxes[dC-1, :, :], dtype=np.float32)
                                est_points = np.ascontiguousarray(np.asarray(detPoses[i], dtype=np.float32).T, dtype=np.float32).reshape(
                                        (8, 1, 2))

                                K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
                                retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                              imagePoints=est_points, cameraMatrix=K,
                                                                              distCoeffs=None, rvec=None, tvec=itvec, useExtrinsicGuess=True, iterationsCount=100,
                                                                              reprojectionError=8.0, confidence=0.99,
                                                                              flags=cv2.SOLVEPNP_ITERATIVE)
                                rmat, _ = cv2.Rodrigues(orvec)
                                #rd = re(np.array(gtRots[j], dtype=np.float32).reshape(3, 3), rmat)
                                #xyz = te((np.array(gtPoses[j], dtype=np.float32)*0.001), (otvec.T))
                                print('--------------------- ICP refinement -------------------')

                                pcd_img = create_point_cloud(image_dep, fxkin, fykin, cxkin, cykin, 0.001)
                                pcd_img = pcd_img.reshape((480, 640, 3))[int(b1[1]):int(b1[3]), int(b1[0]):int(b1[2]), :]
                                pcd_img = pcd_img.reshape((pcd_img.shape[0]*pcd_img.shape[1], 3))
                                pcd_crop = open3d.PointCloud()
                                pcd_crop.points = open3d.Vector3dVector(pcd_img)
                                open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(
                                    radius=0.02, max_nn=30))

                                #pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                                #open3d.draw_geometries([pcd_crop])

                                guess = np.zeros((4, 4), dtype=np.float32)
                                guess[:3, :3] = rmat
                                guess[:3, 3] = itvec.T
                                guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

                                reg_p2p =open3d.registration_icp(pcd_model, pcd_crop, 0.015, guess, open3d.TransformationEstimationPointToPlane())
                                R_est = reg_p2p.transformation[:3, :3]
                                t_est = reg_p2p.transformation[:3, 3]

                                print('--------------------- Evaluate on Metrics -------------------')
                                R_gt = np.array(gtRots[j], dtype=np.float32).reshape(3, 3)
                                t_gt = np.array(gtPoses[j], dtype=np.float32) * 0.001

                                rd = re(R_gt, R_est)
                                xyz = te(t_gt, t_est.T)

                                rot = tf3d.quaternions.mat2quat(R_est)
                                pose = np.concatenate((np.array(otvec[:,0], dtype=np.float32), np.array(rot, dtype=np.float32)), axis=0)
                                posevis.append(pose)

                                if not math.isnan(rd):
                                    rotD.append(rd)
                                    if rd < 5.0 and xyz < 0.05:
                                        less5.append(rd)

                                #if not math.isnan(xyz):
                                #    xyzD.append(xyz)
                                #    if xyz < 0.05:
                                #        less5cm.append(xyz)

                                #model_pts = np.asarray(pcd_model.points)

                                err_vsd = vsd(R_est, t_est * 1000.0, R_gt, t_gt * 1000.0, model_vsd_mm, image_dep, K, 0.3, 20.0)
                                print('vsd: ', err_vsd)
                                if not math.isnan(err_vsd):
                                    vsd_e.append(err_vsd)
                                    if err_vsd < 0.3:
                                        vsd_less_t.append(err_vsd)

                                err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                                print('repr: ', err_repr)

                                if not math.isnan(err_repr):
                                    rep_e.append(err_repr)
                                    if err_repr < 5.0:
                                        rep_less5.append(err_repr)

                                if dC == 3 or dC == 7 or dC == 10 or dC == 11:
                                    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                                else:
                                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                                print('3Dpose: ', err_add, 'th: ', model_radii[dC-1]*2*0.1)

                                if not math.isnan(err_add):
                                    add_e.append(err_add)
                                    if err_add < (model_radii[dC-1]*2*0.1):
                                        add_less_d.append(err_add)

                            else:
                                falsePos.append(dC)


                            #else:
                            #    falsePos.append(dC)

                    fp = falsePos
                    tp = truePos
                    fn += fnCur

                gtCatLst[s] = gtCatLst[s] + 1
                detCatLst[s] = detCatLst[s] + len(tp)
                #detCatLst[s] = detCatLst[s] + tp
                falsePosLst[s] = falsePosLst[s] + len(fp)
                #falseNegLst[s] = falseNegLst[s] + fn
                #print(falseNegLst[s])

                # VISUALIZATION
                if visu == True:

                    #for i, categ in enumerate(gtCats):
                    #    print('gt: ', categ, '::::::', gtPoses[i])

                    img = cv2.imread(rgbImgPath, -1)
                    img_gt = copy.deepcopy(img)
                    #for i, bb in enumerate(detBoxes):
                    #for i, bb in enumerate(gtBoxes):
                        #print(bb)
                        #cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                        #                  (0, 0, 0), 3)
                        #cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                        #                  (250, 215, 10), 2)
                        #if i==7 or i==2:
                        #    cv2.rectangle(img, (int(bb[0]), int(bb[1])),
                        #                  (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                        #                  (0, 0, 0), 3)
                        #    cv2.rectangle(img, (int(bb[0]), int(bb[1])),
                        #                  (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                        #                  (15, 75, 250), 2)

                    for i, bb in enumerate(detBoxes):

                        #segm = detSeg[i]
                        #maskori = cocomask.decode(segm)
                        #mask = np.repeat(maskori[:, :, np.newaxis], 3, axis=2)
                        #r = int(np.random.uniform(100, 255))
                        #g = int(np.random.uniform(100, 255))
                        #b = int(np.random.uniform(100, 255))
                        # mask[:,:,0] *= r
                        # mask[:, :, 1] *= g
                        # mask[:, :, 2] *= b
                        #img[:, :, 0] = np.where(maskori == 1, r, img[:, :, 0])
                        #img[:, :, 1] = np.where(maskori == 1, g, img[:, :, 1])
                        #img[:, :, 2] = np.where(maskori == 1, b, img[:, :, 2])

                        pose2D = detPoses[i]
                        # print(str(cats[i]))
                        pose = np.asarray(detPoses[i], dtype=np.float32)
                        #colR = np.random.uniform(0, 255)
                        #colG = np.random.uniform(0, 255)
                        #colB = np.random.uniform(0, 255)

                        colR = 249
                        colG = 119
                        colB = 25

                        colR1 = 149
                        colG1 = 119
                        colB1 = 179

                        colR2 = 190
                        colG2 = 78
                        colB2 = 194

                        colR3 = 61
                        colG3 = 207
                        colB3 = 194

                        colR4 = 64
                        colG4 = 12
                        colB4 = 194

                        colR5 = 111
                        colG5 = 78
                        colB5 = 246

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (colR, colG, colB), 3)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (colR, colG, colB), 3)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (colR1, colG1, colB1), 3)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (colR1, colG1, colB1), 3)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR2, colG2, colB2), 3)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR2, colG2, colB2), 3)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR5, colG5, colB5), 3)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR5, colG5, colB5), 3)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), (colR3, colG3, colB3),
                                       3)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), (colR3, colG3, colB3),
                                       3)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), (colR4, colG4, colB4),
                                       3)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), (colR4, colG4, colB4),
                                       3)



                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (int(bb[0])+5, int(bb[1])+int(bb[3])-5)
                        fontScale = 0.5
                        fontColor = (25, 215, 250)
                        fontthickness = 2
                        lineType = 2

                        if dataset == 'linemod':
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
                        else:
                            gtText = str(detCats[i])
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


                        #R = np.asarray(gtRots[i], dtype=np.float32)
                        #rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                        #tra = np.asarray(gtPoses[i], dtype=np.float32) * 0.001
                        #xpix = ((tra[0] * fxkin) / tra[2]) + cxkin
                        #ypix = ((tra[1] * fykin) / tra[2]) + cykin
                        #zpix = tra[2] * 0.001 * fxkin
                        #pose = np.concatenate([tra.transpose(), rot.transpose()])
                        #print('gt rotation: ', rot)
                        #draw_axis(img_gt, pose)
                        '''
                        rot = np.asarray(detPoses[i], dtype=np.float32)
                        pose = np.concatenate([tra.transpose(), rot.transpose()])
                        pose = [np.asscalar(pose[0]), np.asscalar(pose[1]), np.asscalar(pose[2]),
                                np.asscalar(pose[3]), np.asscalar(pose[4]), np.asscalar(pose[5]),
                                np.asscalar(pose[6])]
                        pose = np.asarray(pose, dtype=np.float32)
                        #print('det rotation: ', rot)
                        #print('rotation difference: ', dRot)
                        draw_axis(img, pose)
                       # cv2.circle(img, (int(detPoses[i][0]), int(detPoses[i][1])), 5, (255,0,0), 3)
                        #cv2.circle(img, (int(xpix), int(ypix)), 5, (0, 255, 0), 3)
                        '''
                        #rot = np.asarray(gtRots[0], dtype=np.float32).reshape((3,3))
                        #tra = np.asarray(gtPoses[0], dtype=np.float32) * 0.001

                        #tDbox = rot.dot(threeD_boxes[gtCats[0]-1, :, :].T).T
                        #tDbox = tDbox + np.repeat(tra[:, np.newaxis], 8, axis=1).T

                        #box3D = toPix_array(tDbox)
                        #box3D = np.reshape(box3D, (16))
                        #pose = box3D



                        #img_gt = cv2.line(img_gt, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (colR, colG, colB), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (colR, colG, colB), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (colR1, colG1, colB1), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (colR1, colG1, colB1), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR2, colG2, colB2), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR2, colG2, colB2), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR5, colG5, colB5), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR5, colG5, colB5), 3)
                        #img_gt = cv2.line(img_gt, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), (colR3, colG3, colB3),
                        #               3)
                        #img_gt = cv2.line(img_gt, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), (colR3, colG3, colB3),
                        #               3)
                        #img_gt = cv2.line(img_gt, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), (colR4, colG4, colB4),
                        #               3)
                        #img_gt = cv2.line(img_gt, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), (colR4, colG4, colB4),
                        #               3)

                        #draw_axis(img, posevis[0])

                    #name = '/home/sthalham/visTests/detected.jpg'
                    #img_con = np.concatenate((img, img_gt), axis=1)
                    #cv2.imwrite(name, img_con)
                    name_est = '/home/sthalham/visTests/detected_est.jpg'
                    cv2.imwrite(name_est, img)

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




