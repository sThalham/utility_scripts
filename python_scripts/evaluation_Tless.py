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



#Input arguments:
#pcd_temp_ = point cloud(open3d format) of a rendered object in the estimated pose, tf
#pcd_scene_ =point cloud(open3d format) of a scene with surface normals (cropped from the ROI)
#  - both pcd_temp_, pcd_scene_ are downsampled, voxel_size=0.005
#    pcd_scene_ = voxel_down_sample(pcd_scene, voxel_size = 0.005)
# - pcd_scene_ needs surface normals for PointToPlane ICP
#   estimate_normals(pcd_scene_, search_param = KDTreeSearchParamHybrid(
#                                 radius = 0.01, max_nn = 5))

#inlier_thres = max_diameter (of each individual object)
#tf = current pose estimation (4x4 matrix)
#final_th = 0, (this is only used when you need fitness values with a particular threshold)

#Output values:
#[1] tf : transformation matrix after ICP
#[2] inlier_rmse (mean value of inliers after the final icp)
#[3] tf_pcd (point cloud after ICP)
#[4] reg_p2p.fitness (fitness value)


def get_evaluation(pcd_temp_,pcd_scene_,inlier_thres,tf,final_th,n_iter=5):#queue
    tf_pcd =np.eye(4)

    reg_p2p = registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
              TransformationEstimationPointToPoint(),
              ICPConvergenceCriteria(max_iteration = 1)) #5?
    tf = np.matmul(reg_p2p.transformation,tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    for i in range(4):
        inlier_thres = reg_p2p.inlier_rmse*2
        reg_p2p = registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                  TransformationEstimationPointToPlane(),
                  ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation,tf)
        tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)
    inlier_rmse = reg_p2p.inlier_rmse

    ##Calculate fitness with depth_inlier_th
    if(final_th>0):
        inlier_thres = final_th #depth_inlier_th*2 #reg_p2p.inlier_rmse*3
        reg_p2p = registration_icp(pcd_temp_,pcd_scene_, inlier_thres, np.eye(4),
                  TransformationEstimationPointToPlane(),
                  ICPConvergenceCriteria(max_iteration = 1)) #5?

    if( np.abs(np.linalg.det(tf[:3,:3])-1)>0.001):
        tf[:3,0]=tf[:3,0]/np.linalg.norm(tf[:3,0])
        tf[:3,1]=tf[:3,1]/np.linalg.norm(tf[:3,1])
        tf[:3,2]=tf[:3,2]/np.linalg.norm(tf[:3,2])
    if( np.linalg.det(tf) < 0) :
        tf[:3,2]=-tf[:3,2]

    return tf,inlier_rmse,tf_pcd,reg_p2p.fitness


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


def load_pcd(cat):
    # load meshes
    mesh_path = "/home/stefan/data/Meshes/tless_BOP/"
    #mesh_path = "/home/stefan/data/val_linemod_cc_rgb/models_ply/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    pcd_model = open3d.read_point_cloud(ply_path)

    return pcd_model, model_vsd, model_vsd_mm


if __name__ == "__main__":

    root = '/home/stefan/'
    jsons = root + 'tless_bbox_results.json'
    model_path = "/home/stefan/data/Meshes/tless_BOP"
    testData = '/home/stefan/data/train_data/val_tless_dep/'
    annos = testData + 'annotations/instances_val.json'
    mesh_info = '/home/stefan/data/Meshes/tless_BOP/models_info.json'

    visu = True

    json_data = open(jsons).read()
    data = json.loads(json_data)

    json_annos = open(annos).read()
    anno_dict = json.loads(json_annos)

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)

    for key, value in yaml.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x'] * fac
        y_minus = value['min_y'] * fac
        z_minus = value['min_z'] * fac
        x_plus = value['size_x'] * fac + x_minus
        y_plus = value['size_y'] * fac + y_minus
        z_plus = value['size_z'] * fac + z_minus
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                  [x_plus, y_plus, z_minus],
                                  [x_plus, y_minus, z_minus],
                                  [x_plus, y_minus, z_plus],
                                  [x_minus, y_plus, z_plus],
                                  [x_minus, y_plus, z_minus],
                                  [x_minus, y_minus, z_minus],
                                  [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter']


    gt_cls = np.zeros((31,), dtype=np.int32)
    det_cls = np.zeros((31,), dtype=np.int32)
    vsd_e = np.zeros((31,), dtype=np.int32)
    vsd_less_t = np.zeros((31,), dtype=np.int32)

    # target annotation
    pc1, mv1, mv1_mm = load_pcd('01')
    pc2, mv2, mv2_mm = load_pcd('02')
    pc3, mv3, mv3_mm = load_pcd('03')
    pc4, mv4, mv4_mm = load_pcd('04')
    pc5, mv5, mv5_mm = load_pcd('05')
    pc6, mv6, mv6_mm = load_pcd('06')
    pc7, mv7, mv7_mm = load_pcd('07')
    pc8, mv8, mv8_mm = load_pcd('08')
    pc9, mv9, mv9_mm = load_pcd('09')
    pc10, mv10, mv10_mm = load_pcd('10')
    pc11, mv11, mv11_mm = load_pcd('11')
    pc12, mv12, mv12_mm = load_pcd('12')
    pc13, mv13, mv13_mm = load_pcd('13')
    pc14, mv14, mv14_mm = load_pcd('14')
    pc15, mv15, mv15_mm = load_pcd('15')
    pc16, mv16, mv16_mm = load_pcd('16')
    pc17, mv17, mv17_mm = load_pcd('17')
    pc18, mv18, mv18_mm = load_pcd('18')
    pc19, mv19, mv19_mm = load_pcd('19')
    pc20, mv20, mv20_mm = load_pcd('20')
    pc21, mv21, mv21_mm = load_pcd('21')
    pc22, mv22, mv22_mm = load_pcd('22')
    pc23, mv23, mv23_mm = load_pcd('23')
    pc24, mv24, mv24_mm = load_pcd('24')
    pc25, mv25, mv25_mm = load_pcd('25')
    pc26, mv26, mv26_mm = load_pcd('26')
    pc27, mv27, mv27_mm = load_pcd('27')
    pc28, mv28, mv28_mm = load_pcd('28')
    pc29, mv29, mv29_mm = load_pcd('29')
    pc30, mv30, mv30_mm = load_pcd('30')

    for gt in anno_dict['annotations']:

        cls_gt = int(gt['category_id'])
        calib_gt = gt['calib']
        box_gt = gt['bbox']
        pose_gt = gt['pose']

        gt_cls[cls_gt] += 1
        vsd_e[cls_gt] += 1
        det_true = False
        vsd_true = False

        dets = []
        for det in data:
            if det['image_id'] == gt['image_id']:
                dets.append(det)

        for item in dets:
            cls_det = item['category_id']
            if cls_det == cls_gt:
                box_det = item['bbox']
                IoU = boxoverlap(box_gt, box_det)
                if IoU > 0.5:
                    det_cls[cls_det] += 1
                    box3D_det = item['pose']

                    obj_points = np.ascontiguousarray(threeD_boxes[cls_det - 1, :, :],
                                                      dtype=np.float32)  # .reshape((8, 1, 3))
                    est_points = np.ascontiguousarray(box3D_det, dtype=np.float32).reshape((8, 1, 2))

                    calib = calib_gt
                    K = np.float32([calib[0], 0., calib[2], 0., calib[1], calib[3], 0., 0., 1.]).reshape(3, 3)

                    # retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                    retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                       imagePoints=est_points, cameraMatrix=K,
                                                                       distCoeffs=None, rvec=None, tvec=None,
                                                                       useExtrinsicGuess=False, iterationsCount=100,
                                                                       reprojectionError=5.0, confidence=0.99,
                                                                       flags=cv2.SOLVEPNP_ITERATIVE)

                    R_est, _ = cv2.Rodrigues(orvec)
                    t_est = otvec

                    R_gt = pose_gt[3:]
                    R_gt = tf3d.euler.euler2mat(R_gt[0], R_gt[1], R_gt[2])
                    R_gt = np.array(R_gt, dtype=np.float32).reshape(3,3)
                    t_gt = pose_gt[:3]
                    t_gt = np.array(t_gt, dtype=np.float32) * 0.001

                    name_template = '00000'
                    len_img_id = len(str(gt['image_id']))
                    img_id = name_template[:-len_img_id] + str(gt['image_id'])
                    depImgPath = testData + 'images/val/' + img_id + '_dep.png'
                    image_dep = cv2.imread(depImgPath, -1)
                    '''
                    print('--------------------- ICP refinement -------------------')
                    image_dep = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
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
                    '''

                    cls = cls_det
                    if cls == 1:
                        model_vsd = mv1
                    elif cls == 2:
                        model_vsd = mv2
                    elif cls == 3:
                        model_vsd = mv3
                    elif cls == 4:
                        model_vsd = mv4
                    elif cls == 5:
                        model_vsd = mv5
                    elif cls == 6:
                        model_vsd = mv6
                    elif cls == 7:
                        model_vsd = mv7
                    elif cls == 8:
                        model_vsd = mv8
                    elif cls == 9:
                        model_vsd = mv9
                    elif cls == 10:
                        model_vsd = mv10
                    elif cls == 11:
                        model_vsd = mv11
                    elif cls == 12:
                        model_vsd = mv12
                    elif cls == 13:
                        model_vsd = mv13
                    elif cls == 14:
                        model_vsd = mv14
                    elif cls == 15:
                        model_vsd = mv15
                    elif cls == 16:
                        model_vsd = mv16
                    elif cls == 17:
                        model_vsd = mv17
                    elif cls == 18:
                        model_vsd = mv18
                    elif cls == 19:
                        model_vsd = mv19
                    elif cls == 20:
                        model_vsd = mv20
                    elif cls == 21:
                        model_vsd = mv21
                    elif cls == 22:
                        model_vsd = mv22
                    elif cls == 23:
                        model_vsd = mv23
                    elif cls == 24:
                        model_vsd = mv24
                    elif cls == 25:
                        model_vsd = mv25
                    elif cls == 26:
                        model_vsd = mv26
                    elif cls == 27:
                        model_vsd = mv27
                    elif cls == 28:
                        model_vsd = mv28
                    elif cls == 29:
                        model_vsd = mv29
                    elif cls == 30:
                        model_vsd = mv30

                    print(depImgPath)
                    print(np.amax(image_dep))
                    err_vsd = vsd(R_est, t_est * 1000.0, R_gt, t_gt * 1000.0, model_vsd, image_dep, K, 0.3, 20.0)
                    print('vsd: ', err_vsd)
                    if not math.isnan(err_vsd):
                        if err_vsd < 0.3:
                            vsd_less_t[cls_gt] += 1

    Rec_avg = 0
    vsd_avg = 0
    for ind in range(1, 31):
        if ind == 0:
            continue

        else:
            Rec = det_cls[ind] / gt_cls[ind]
            Rec_avg += Rec
            vsd_score = vsd_less_t[ind] / vsd_e[ind]
            vsd_avg += vsd_score
            print('CLS: ', ind, 'Recall: ', Rec, 'VSD: ', vsd_score)

    print('Overall: ', 'Recall: ', Rec_avg/30, 'VSD: ', vsd_avg/30)



