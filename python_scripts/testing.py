import sys
import os
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import itertools
#import pcl
import scipy
#from pcl import pcl_visualization
import OpenEXR, Imath
import segment_normals as seg
import time


resX = 640
resY = 480
kin_res_x = 720
kin_res_y = 540
fov = 57.8
focalLx = 579.68  # blender calculated
focalLy = 542.31  # blender calculated
depthCut = 1500.0

np.set_printoptions(threshold=np.nan)


def HHA_encoding_approx(depth, normals, fx, fy, cx, cy, ds,):

    r, c, p = normals.shape
    mask = depth < depthCut
    normals[:, :, 0] = np.where(mask, normals[:, :, 0], np.NaN)
    normals[:, :, 1] = np.where(mask, normals[:, :, 1], np.NaN)
    normals[:, :, 2] = np.where(mask, normals[:, :, 2], np.NaN)

    # calculate disparity
    depthFloor = 100.0
    depthCeil = depthCut

    disparity = np.ones((depth.shape), dtype=np.float32)
    disparity = np.divide(disparity, depth)
    disparity = disparity - (1 / depthCeil)
    denom = (1 / depthFloor) - (1 / depthCeil)
    disparity = np.divide(disparity, denom)
    disparity = np.where(np.isinf(disparity), 0.0, disparity)
    dispSca = disparity - np.nanmin(disparity)
    maxV = 255.0 / np.nanmax(dispSca)
    scatemp = np.multiply(dispSca, maxV)
    disp_final = scatemp.astype(np.uint8)
    disp_final = np.where(mask, disp_final, 0)

    # compute gravity vector deviation
    angEst = np.zeros(normals.shape, dtype=np.float32)
    angEst[:, :, 2] = 1.0
    ang = (45.0, 45.0, 45.0, 45.0, 45.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0)
    for th in ang:
        angtemp = np.einsum('ijk,ijk->ij', normals, angEst)
        angEstNorm = np.linalg.norm(angEst, axis=2)
        normalsNorm = np.linalg.norm(normals, axis=2)
        normalize = np.multiply(normalsNorm, angEstNorm)
        angDif = np.divide(angtemp, normalize)

        np.where(angDif < 0.0, angDif + 1.0, angDif)
        angDif = np.arccos(angDif)
        angDif = np.multiply(angDif, (180 / math.pi))

        cond1 = (angDif < th)
        cond1_ = (angDif > (180.0 - th))
        cond2 = (angDif > (90.0 - th)) & (angDif < (90.0 + th))
        cond1 = np.repeat(cond1[:, :, np.newaxis], 3, axis=2)
        cond1_ = np.repeat(cond1_[:, :, np.newaxis], 3, axis=2)
        cond2 = np.repeat(cond2[:, :, np.newaxis], 3, axis=2)

        NyPar1 = np.extract(cond1, normals)
        NyPar2 = np.extract(cond1_, normals)
        NyPar = np.concatenate((NyPar1, NyPar2))
        npdim = (NyPar.shape[0] / 3)
        NyPar = np.reshape(NyPar, (int(npdim), 3))
        NyOrt = np.extract(cond2, normals)
        nodim = (NyOrt.shape[0] / 3)
        NyOrt = np.reshape(NyOrt, (int(nodim), 3))

        cov = (np.transpose(NyOrt)).dot(NyOrt) - (np.transpose(NyPar)).dot(NyPar)
        u, s, vh = np.linalg.svd(cov)
        angEst = np.tile(u[:, 2], r * c).reshape((r, c, 3))

    angDifSca = angDif - np.nanmin(angDif)
    maxV = 255.0 / np.nanmax(angDifSca)
    scatemp = np.multiply(angDifSca, maxV)
    gImg = scatemp.astype(np.uint8)
    gImg[gImg is np.NaN] = 0

    # encode
    encoded = np.zeros((normals.shape), dtype=np.uint8)
    encoded[:, :, 0] = disp_final
    encoded[:, :, 1] = height_final
    encoded[:, :, 2] = grav_final

    return encoded


def HHA_encoding(depth, normals, fx, fy, cx, cy, ds, R_w2c, T_w2c):

    rows, cols, channels = normals.shape
    mask = depth < depthCut

    # calculate disparity
    depthFloor = 100.0
    depthCeil = depthCut

    disparity = np.ones((depth.shape), dtype=np.float32)
    disparity = np.divide(disparity, depth)
    disparity = disparity - (1 / depthCeil)
    denom = (1 / depthFloor) - (1 / depthCeil)
    disparity = np.divide(disparity, denom)
    disparity = np.where(np.isinf(disparity), 0.0, disparity)
    dispSca = disparity - np.nanmin(disparity)
    maxV = 255.0 / np.nanmax(dispSca)
    scatemp = np.multiply(dispSca, maxV)
    disp_final = scatemp.astype(np.uint8)
    disp_final = np.where(mask, disp_final, 0)

    # compute height
    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)
    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud = np.transpose(np.array((xP, yP, zP)))

    zFloor = 0.0
    zCeil = 1000

    R_cam = np.asarray(R_w2c).reshape(3, 3)
    camPoints = np.transpose(np.matmul(R_cam, np.transpose(cloud))) + np.tile(T_w2c, cloud.shape[0]).reshape(
        cloud.shape[0], 3)
    height = camPoints[:, 2] - zFloor
    denom = zCeil - zFloor
    height = np.divide(height, denom)
    height = height.reshape(rows, cols)
    height = height - np.nanmin(height)
    scatemp = np.multiply(height, 255.0)
    height_final = scatemp.astype(np.uint8)
    height_final = np.where(height_final <= 0.0, 0.0, height_final)
    height_final = np.where(mask, height_final, 0)

    # compute gravity vector deviation
    angEst = np.zeros(normals.shape, dtype=np.float32)
    angEst[:, :, 0] = R_cam[0, 2]
    angEst[:, :, 1] = R_cam[1, 2]
    angEst[:, :, 2] = -R_cam[2, 2]

    angtemp = np.einsum('ijk,ijk->ij', normals, angEst)
    angEstNorm = np.linalg.norm(angEst, axis=2)
    normalsNorm = np.linalg.norm(normals, axis=2)
    normalize = np.multiply(normalsNorm, angEstNorm)
    angDif = np.divide(angtemp, normalize)

    np.where(angDif < 0.0, angDif + 1.0, angDif)
    angDif = np.arccos(angDif)
    angDif = np.multiply(angDif, (180 / math.pi))

    angDif = np.where(mask, angDif, np.NaN)
    angDifSca = angDif - np.nanmin(angDif)
    maxV = 255.0 / np.nanmax(angDifSca)
    scatemp = np.multiply(angDifSca, maxV)
    grav_final = scatemp.astype(np.uint8)
    grav_final[grav_final is np.NaN] = 0
    grav_final = np.where(mask, grav_final, 0)

    # encode
    encoded = np.zeros((normals.shape), dtype=np.uint8)
    encoded[:, :, 0] = disp_final
    encoded[:, :, 1] = height_final
    encoded[:, :, 2] = grav_final

    return encoded


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

    return cloud_final


def compute_height(cloud, y, x, R_w2c, T_w2c):

    zFloor = 0.0
    zCeil = 1000

    cam2world = R_w2c
    camPoints = np.transpose(np.matmul(cam2world, np.transpose(cloud))) + np.tile(T_w2c, cloud.shape[0]).reshape(cloud.shape[0], 3)

    height = camPoints[:, 2] - zFloor
    denom = zCeil - zFloor
    height = np.divide(height, denom)

    height = np.where(height <= 0.0, 0.0, height)
    height = height.reshape(y, x)

    height = height - np.nanmin(height)
    scatemp = np.multiply(height, 255.0)
    height_final = scatemp.astype(np.uint8)

    return height_final


def comp_disp(depth):

    depthFloor = 100.0
    depthCeil = 1000.0

    disparity = np.ones((depth.shape), dtype=np.float32)
    print(disparity.shape)
    disparity = np.divide(disparity, depth)
    disparity = disparity - (1/depthCeil)
    denom = (1/depthFloor) - (1/depthCeil)
    disparity = np.divide(disparity, denom)
    disparity = np.where(np.isinf(disparity), 0.0, disparity)

    dispSca = disparity - np.nanmin(disparity)
    maxV = 255.0 / np.nanmax(dispSca)
    scatemp = np.multiply(dispSca, maxV)
    disp_final = scatemp.astype(np.uint8)

    return disp_final


def geoCooFrame(normals, depth):

    r, c, p = normals.shape
    mask = depth < depthCut
    normals[:, :, 0] = np.where(mask, normals[:, :, 0], np.NaN)
    normals[:, :, 1] = np.where(mask, normals[:, :, 1], np.NaN)
    normals[:, :, 2] = np.where(mask, normals[:, :, 2], np.NaN)

    angEst = np.zeros(normals.shape, dtype=np.float32)
    angEst[:, :, 2] = 1.0
    ang = (65.0, 65.0, 45.0, 45.0, 45.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0)
    for th in ang:

        angtemp = np.einsum('ijk,ijk->ij', normals, angEst)
        angEstNorm = np.linalg.norm(angEst, axis=2)
        normalsNorm = np.linalg.norm(normals, axis=2)
        normalize = np.multiply(normalsNorm, angEstNorm)
        angDif = np.divide(angtemp, normalize)

        np.where(angDif < 0.0, angDif + 1.0, angDif)
        angDif = np.arccos(angDif)
        angDif = np.multiply(angDif, (180/math.pi))

        cond1 = (angDif < th)
        cond1_ = (angDif > (180.0 - th))
        cond2 = (angDif > (90.0 - th)) & (angDif < (90.0 + th))
        cond1 = np.repeat(cond1[:, :, np.newaxis], 3, axis=2)
        cond1_ = np.repeat(cond1_[:, :, np.newaxis], 3, axis=2)
        cond2 = np.repeat(cond2[:, :, np.newaxis], 3, axis=2)

        NyPar1 = np.extract(cond1, normals)
        NyPar2 = np.extract(cond1_, normals)
        NyPar = np.concatenate((NyPar1, NyPar2))
        npdim = (NyPar.shape[0] / 3)
        NyPar = np.reshape(NyPar, (int(npdim), 3))
        NyOrt = np.extract(cond2, normals)
        nodim = (NyOrt.shape[0] / 3)
        NyOrt = np.reshape(NyOrt, (int(nodim), 3))

        cov = (np.transpose(NyOrt)).dot(NyOrt) - (np.transpose(NyPar)).dot(NyPar)
        u, s, vh = np.linalg.svd(cov)
        angEst = np.tile(u[:, 2], r * c).reshape((r, c, 3))

    angDifSca = angDif - np.nanmin(angDif)
    maxV = 255.0 / np.nanmax(angDifSca)
    scatemp = np.multiply(angDifSca, maxV)
    gImg = scatemp.astype(np.uint8)
    gImg[gImg is np.NaN] = 0

    return gImg


def roundPartial(value, resolution):
    if resolution == 0:
        return 0
    return round(value / resolution) * resolution


def manipulate_depth(fn_gt, fn_depth, fn_part, fn_mask):

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        bboxes = np.zeros((len(query), 5), np.int)
        poses = np.zeros((len(query), 7), np.float32)
        mask_ids = np.zeros((len(query)), np.int)
        for j in range(len(query)-1):
            qr = query[j]
            class_id = qr['class_id']
            bbox = qr['bbox']
            mask_ids[j] = int(qr['mask_id'])
            pose = np.array(qr['pose']).reshape(4, 4)
            bboxes[j, 0] = class_id
            bboxes[j, 1:5] = np.array(bbox)
            q_pose = tf3d.quaternions.mat2quat(pose[:3, :3])
            poses[j, :4] = np.array(q_pose)
            poses[j, 4:7] = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(fn_depth)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depth = np.fromstring(redstr, dtype=np.float32)
    depth.shape = (size[1], size[0])

    centerX = kin_res_x / 2.0
    centerY = kin_res_y / 2.0

    uv_table = np.zeros((kin_res_y, kin_res_x, 2), dtype=np.int16)
    column = np.arange(0, kin_res_y)
    uv_table[:, :, 1] = np.arange(0, kin_res_x) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / kin_res_x * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / kin_res_x * uv_table[:, :, 0]))

    depth = cv2.resize(depth, (resX, resY))

    # erode and blur mask to get more realistic appearance
    partmask = cv2.imread(fn_part, 0)
    partmask = cv2.resize(partmask, (resX, resY))
    partmask = partmask.astype(np.float32)
    mask = partmask > (np.median(partmask)*0.4)
    partmask = np.where(mask, 255.0, 0.0)
    ###################################################
    # DON'T REMOVE, best normal map up to now !!! tested without lateral noise !!!
    kernel = np.ones((7, 7))
    #partmask = signal.medfilt2d(partmask, kernel_size=7)
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=3)
    ###################################################
    partmask = partmask.astype(np.uint8)

    scaDep = 255.0 / np.nanmax(partmask)
    depImg = np.multiply(partmask, scaDep)
    depI = depImg.astype(np.uint8)
    cv2.imwrite("/home/sthalham/visTests/mask.jpg", depI)

    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    scaDep = 255.0 / np.nanmax(depth)
    depImg = np.multiply(depth, scaDep)
    depI = depImg.astype(np.uint8)
    cv2.imwrite("/home/sthalham/visTests/depMasked.jpg", depI)

    onethird = cv2.resize(depth, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)
    res = (((onethird / 1000) * 1.41421356) ** 2) * 1000
    depth = onethird

    # discretize to resolution and apply gaussian
    dNonVar = np.divide(depth, res, out=np.zeros_like(depth), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)
    noise = np.multiply(dNonVar, 0.0025)
    depthFinal = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)

    depthFinal = cv2.GaussianBlur(depthFinal, (7, 7), 0.75, 0.75)

    # apply retardo-noise to non-object parts
    objmask = np.load(fn_mask)
    objmask = cv2.resize(objmask, (720, 540))
    objmask = np.where(objmask > 0, 255, 0)
    print(objmask.shape)
    print(np.nanmax(objmask))
    print(np.nanmin(objmask))
    edges = cv2.Canny(objmask, 100, 200)

    cv2.imwrite('/home/sthalham/visTests/edges.jpg', edges)

    # INTER_NEAREST - a nearest-neighbor interpolation
    # INTER_LINEAR - a bilinear interpolation (used by default)
    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    depthFinal = cv2.resize(depthFinal, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    return depthFinal, bboxes, poses, mask_ids


def get_normal(depth_refine, fx=-1.0, fy=-1.0, cx=-1, cy=-1, for_vis=True):
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]

    # inpainting

    scaleOri = np.amax(depth_refine)
    inPaiMa = np.where(depth_refine == 0.0, 255, 0)
    inPaiMa = inPaiMa.astype(np.uint8)
    inPaiDia = 5.0
    depth_refine = depth_refine.astype(np.float32)
    depPaint = cv2.inpaint(depth_refine, inPaiMa, inPaiDia, cv2.INPAINT_NS)

    depNorm = depPaint - np.amin(depPaint)
    rangeD = np.amax(depNorm)
    depNorm = np.divide(depNorm, rangeD)
    depth_refine = np.multiply(depNorm, scaleOri)


    centerX = cx
    centerY = cy

    constant = 1 / fx
    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)

    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)
    uv_table_sign = np.copy(uv_table)

    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)

    v_x = np.zeros((res_y, res_x, 3))
    v_y = np.zeros((res_y, res_x, 3))
    normals = np.zeros((res_y, res_x, 3))

    dig = np.gradient(depth_refine, 3, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant * dig[0]
    v_y[:, :, 1] = depth_refine * constant + (uv_table_sign[:, :, 0] * constant) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant + uv_table_sign[:, :, 1] * constant * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)

    cross = cross / norm
    cross = cross.reshape(res_y, res_x, 3)
    cross = np.abs(cross)
    cross = np.nan_to_num(cross)

    cam_angle = np.arccos(cross[:, :, 2])
    #cross[np.abs(cam_angle) > math.radians(75)] = 0  # high normal cut
    #cross[depth_refine <= 100] = 0  # 0 and near range cut
    cross[depth_refine > 1500] = np.NaN  # far range cut
    if not for_vis:
        scaDep = 1.0/np.nanmax(depth_refine)
        depth_norm = np.multiply(depth_refine, scaDep)
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_norm - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_norm - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_norm - 0.5))

    return cross, depth_refine


##########################
#         MAIN           #
##########################
if __name__ == "__main__":

    #root = '/home/sthalham/data/t-less_mani/artificialScenes/renderedScenes03052018'  # path to train samples
    root = '/home/sthalham/data/t-less_mani/artificialScenes/renderedLINEMOD31052018/patches'
    model = '/home/sthalham/workspace/python/src/model.yml.gz'

    #compare1 = cv2.imread('/home/sthalham/data/LINEMOD/train/06/depth/0017.png', -1)

    # linemod
    #fkinx = 572.4114
    #fkiny = 573.57043
    #ckinx = 325.2611
    #ckiny = 242.04899
    #pcA = create_point_cloud(compare1, fkinx, fkiny, ckinx, ckiny, 1.0)
    #cloudA = pcl.PointCloud(np.array(pcA, dtype=np.float32))
    #visual = pcl.pcl_visualization.CloudViewing()
    #visual.ShowMonochromeCloud(cloudA)


    lm_real = cv2.imread('/home/sthalham/data/LINEMOD/test/03/depth/0003.png', -1)
    lm_syn = cv2.imread('/home/sthalham/data/prepro/depLM/coco_train2014/0009089.jpg', -1)
    tl_real = cv2.imread('/home/sthalham/data/t-less_v2/test_primesense/01/depth/0022.png', -1)
    tl_syn = cv2.imread('/home/sthalham/data/prepro/depTL/coco_train2014/0000354.jpg', -1)

    tl_real = np.round(tl_real * 0.1)
    #lm_real[lm_real > 800] = 0
    #tl_real[tl_real > 800] = 0
    #lm_real[lm_real < 700] = 0
    #tl_real[tl_real < 700] = 0
    #print(len(np.unique(lm_real)))
    #print(np.unique(lm_real))
    #print(len(np.unique(tl_real)))
    #print(np.unique(tl_real))

    lm_real[lm_real > 1500] = 0
    tl_syn[tl_syn > 1500] = 0

    #print(np.amax(lm_real.dtype))
    #print(np.amax(lm_syn))
    #print(np.amax(tl_real))
    #print(np.amax(tl_syn))

    cv2.rectangle(lm_syn, (100, 50), (200, 150),
                  (255, 255, 255), 3)
    #cv2.imwrite('/home/sthalham/visTests/lm_syn.png', lm_syn)

    p1 = lm_syn[250:350, 180:280]
    #p1 = cv2.resize(p1, (1000,1000))
    p1 = p1 - np.amin(p1)
    sca = 255.0 / np.amax(p1)
    depth_scaled = np.multiply(p1, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/zlmsyn.png', depth_int8)

    sca = 255.0 / np.amax(lm_real)
    depth_scaled = np.multiply(lm_real, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.rectangle(depth_int8, (100, 280), (200, 380),
                  (255, 255, 255), 3)
    p1 = lm_real[280:380, 100:200]
    #p1 = cv2.resize(p1, (1000, 1000))
    p1 = p1 - np.amin(p1)
    sca = 255.0 / np.amax(p1)
    depth_scaled = np.multiply(p1, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/z_lm_real.png', depth_int8)

    print(np.mean(tl_syn[350:450, 150:250]))
    sca = 255.0 / np.amax(tl_syn)
    depth_scaled = np.multiply(tl_syn, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.rectangle(depth_int8, (150, 250), (250, 350),
                  (255, 255, 255), 3)
    cv2.imwrite('/home/sthalham/visTests/tl_syn.png', depth_int8)
    p1 = tl_syn[350:450, 150:250]
    #p1 = cv2.resize(p1, (1000, 1000))

    p1 = p1 - np.amin(p1)
    sca = 255.0 / np.amax(p1)
    depth_scaled = np.multiply(p1, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/z_tl_syn.png', depth_int8)

    print(np.mean(tl_real[50:250, 50:250]))
    sca = 255.0 / np.amax(tl_real)
    depth_scaled = np.multiply(tl_real, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.rectangle(depth_int8, (50, 50), (150, 150),
                  (255, 255, 255), 3)
    p1 = tl_real[50:150, 50:150]
    #p1 = cv2.resize(p1, (1000, 1000))
    p1 = p1 - np.amin(p1)
    sca = 255.0 / np.amax(p1)
    depth_scaled = np.multiply(p1, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/z_tl_real.png', depth_int8)

    fkinx = 572.4114
    fkiny = 573.57043
    ckinx = 325.2611
    ckiny = 242.04899

    pcA = create_point_cloud(compare1, fkinx, fkiny, ckinx, ckiny, 1.0)
    cloudA = pcl.PointCloud(np.array(pcA, dtype=np.float32))
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowMonochromeCloud(cloudA)

    sca = 255.0 / np.amax(compare1)
    depth_scaled = np.multiply(compare1, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/depth1.png', depth_int8)
    sca = 255.0 / np.amax(compare2)
    depth_scaled = np.multiply(compare2, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/depth2.png', depth_int8)
    sca = 255.0 / np.amax(compare3)
    depth_scaled = np.multiply(compare3, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/depth3.png', depth_int8)
    sca = 255.0 / np.amax(compare4)
    depth_scaled = np.multiply(compare4, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/depth4.png', depth_int8)
    sca = 255.0 / np.amax(compare5)
    depth_scaled = np.multiply(compare5, sca)
    depth_int8 = depth_scaled.astype(np.uint8)
    cv2.imwrite('/home/sthalham/visTests/depth5.png', depth_int8)


    rowcom, colcom = compare1.shape
    # compare1 = np.multiply(compare1, 0.1)
    # compare2 = np.multiply(compare2, 0.1)
    # compare3 = np.multiply(compare3, 0.1)
    # compare4 = np.multiply(compare4, 0.1)
    # compare5 = np.multiply(compare5, 0.1)

    # t-less
    # fkinx = 1076.74
    # fkiny = 1075.18
    cam_R_w2c = np.array([[-0.990781, 0.135477, -0.00023996],[0.0560823, 0.408533, -0.911019], [-0.123324, -0.902633, -0.412365]])
    cam_T_w2c = np.array([-3.62461, -0.291907, 788.726])
    cam_K = np.array([1076.74064739, 0.0, 364.98264967, 0.0, 1075.17825536, 301.59181836, 0.0, 0.0, 1.0])
    # ckinx = 364.98264967
    # ckiny = 301.59181836

    # linemod
    fkinx = 572.4114
    fkiny = 573.57043
    ckinx = 325.2611
    ckiny = 242.04899

    pcA = create_point_cloud(compare1, fkinx, fkiny, ckinx, ckiny, 1.0)
    cloudA = pcl.PointCloud(np.array(pcA, dtype=np.float32))
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowMonochromeCloud(cloudA)

    '''
    normImg1, dptComp1 = get_normal(compare1, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    areaCol = encode_area(dptComp1)
    cv2.imwrite('/home/sthalham/visTests/clusteredReal1.png', areaCol)
    gravImg = geoCooFrame(normImg1, dptComp1)
    cv2.imwrite('/home/sthalham/visTests/gravImg1.png', gravImg)

    normImg2, dptComp2 = get_normal(compare2, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    areaCol = encode_area(dptComp2)
    cv2.imwrite('/home/sthalham/visTests/clusteredReal2.png', areaCol)
    gravImg = geoCooFrame(normImg2, dptComp2)
    cv2.imwrite('/home/sthalham/visTests/gravImg2.png', gravImg)

    normImg3, dptComp3 = get_normal(compare3, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    areaCol = encode_area(dptComp3)
    cv2.imwrite('/home/sthalham/visTests/clusteredReal3.png', areaCol)
    gravImg = geoCooFrame(normImg3, dptComp3)
    cv2.imwrite('/home/sthalham/visTests/gravImg3.png', gravImg)

    normImg4, dptComp4 = get_normal(compare4, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    areaCol = encode_area(dptComp4)
    cv2.imwrite('/home/sthalham/visTests/clusteredReal4.png', areaCol)
    gravImg = geoCooFrame(normImg4, dptComp4)
    cv2.imwrite('/home/sthalham/visTests/gravImg4.png', gravImg)

    normImg5, dptComp5 = get_normal(compare5, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    areaCol = encode_area(dptComp5)
    cv2.imwrite('/home/sthalham/visTests/clusteredReal5.png', areaCol)
    gravImg = geoCooFrame(normImg5, dptComp5)
    cv2.imwrite('/home/sthalham/visTests/gravImg5.png', gravImg)

    
    encoded1 = HHA_encoding_approx(dptComp1, normImg1, fkinx, fkiny, (colcom * 0.5), (rowcom * 0.5), 1.0)
    cv2.imwrite('/home/sthalham/visTests/dispReal1.png', encoded1[:, :, 0])
    cv2.imwrite('/home/sthalham/visTests/heiReal1.png', encoded1[:, :, 1])
    cv2.imwrite('/home/sthalham/visTests/graReal1.png', encoded1[:, :, 2])
    cv2.imwrite('/home/sthalham/visTests/HHAReal1.png', encoded1)

    normImg2, dep2 = get_normal(compare2, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    encoded2 = HHA_encoding_approx(dptComp2, normImg2, fkinx, fkiny, (colcom * 0.5), (rowcom * 0.5), 1.0)
    cv2.imwrite('/home/sthalham/visTests/dispReal2.png', encoded2[:, :, 0])
    cv2.imwrite('/home/sthalham/visTests/heiReal2.png', encoded2[:, :, 1])
    cv2.imwrite('/home/sthalham/visTests/graReal2.png', encoded2[:, :, 2])
    cv2.imwrite('/home/sthalham/visTests/HHAReal2.png', encoded2)

    normImg3, dep3 = get_normal(compare3, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    encoded3 = HHA_encoding_approx(dptComp3, normImg3, fkinx, fkiny, (colcom * 0.5), (rowcom * 0.5), 1.0)
    cv2.imwrite('/home/sthalham/visTests/dispReal3.png', encoded3[:, :, 0])
    cv2.imwrite('/home/sthalham/visTests/heiReal3.png', encoded3[:, :, 1])
    cv2.imwrite('/home/sthalham/visTests/graReal3.png', encoded3[:, :, 2])
    cv2.imwrite('/home/sthalham/visTests/HHAReal3.png', encoded3)

    normImg4, dep4 = get_normal(compare4, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    encoded4 = HHA_encoding_approx(dptComp4, normImg4, fkinx, fkiny, (colcom * 0.5), (rowcom * 0.5), 1.0)
    cv2.imwrite('/home/sthalham/visTests/dispReal4.png', encoded4[:, :, 0])
    cv2.imwrite('/home/sthalham/visTests/heiReal4.png', encoded4[:, :, 1])
    cv2.imwrite('/home/sthalham/visTests/graReal4.png', encoded4[:, :, 2])
    cv2.imwrite('/home/sthalham/visTests/HHAReal4.png', encoded4)

    normImg5, dep5 = get_normal(compare5, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    encoded5 = HHA_encoding_approx(dptComp5, normImg5, fkinx, fkiny, (colcom * 0.5), (rowcom * 0.5), 1.0)
    cv2.imwrite('/home/sthalham/visTests/dispReal5.png', encoded5[:, :, 0])
    cv2.imwrite('/home/sthalham/visTests/heiReal5.png', encoded5[:, :, 1])
    cv2.imwrite('/home/sthalham/visTests/graReal5.png', encoded5[:, :, 2])
    cv2.imwrite('/home/sthalham/visTests/HHAReal5.png', encoded5)
    '''

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
        "description": "tless",
        "url": "cmp.felk.cvut.cz/t-less/",
        "version": "1.0",
        "year": 2018,
        "contributor": "Stefan Thalhammer",
        "date_created": dateT
    },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    annoID = 0
    counter = 0

    depPath = root + "/depth/"
    partPath = root + "/part/"
    gtPath = root
    maskPath = root + "/mask/"

    for fileInd in os.listdir(root):
        if fileInd.endswith(".yaml"):
            redname = fileInd[:-8]
            #print(str(redname))
            #if redname != "00000019":
            #    continue
            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"
            maskfile = maskPath + redname + "_mask.npy"

            with open(gtfile, 'r') as stream:
                query = yaml.load(stream)
                camRot = query['camera_rot']

            depth_refine, bboxes, poses, mask_ids = manipulate_depth(gtfile, depfile, partfile, maskfile)
            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters

            sca = 255.0 / np.amax(depth_refine)
            depth_scaled = np.multiply(depth_refine, sca)
            depth_int8 = depth_scaled.astype(np.uint8)
            cv2.imwrite('/home/sthalham/visTests/depth_refine.png', depth_int8)

            rows, cols = depth_refine.shape

            normImg, depth_inpaint = get_normal(depth_refine, fx=focalLx, fy=focalLy, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5),
                            for_vis=False)
            sca = 255.0 /np.nanmax(normImg)
            normImg = np.multiply(normImg, sca)
            imgI = normImg.astype(np.uint8)
            cv2.imwrite('/home/sthalham/visTests/normarti.png', imgI)

            #pcA = create_point_cloud(depth_refine, focalLx, focalLy, kin_res_x*0.5, kin_res_y*0.5, 1.0)
            #cloudA = pcl.PointCloud(np.array(pcA, dtype=np.float32))
            #visual = pcl.pcl_visualization.CloudViewing()
            #visual.ShowMonochromeCloud(cloudA)

            # pcC = create_point_cloud(compare1, fkinx, fkiny, kin_res_x * 0.5, kin_res_y * 0.5, 1.0)
            # cloudC = pcl.PointCloud(np.array(pcC, dtype=np.float32))
            # visual1 = pcl.pcl_visualization.CloudViewing()
            # visual1.ShowMonochromeCloud(cloudC)



            # encoded = HHA_encoding(depth_refine, normImg, focalLx, focalLy, (kin_res_x * 0.5), (kin_res_y * 0.5), 1.0, R2w, T2w)
            # cv2.imwrite('/home/sthalham/disparity.png', encoded[:, :, 0])
            # cv2.imwrite('/home/sthalham/height.png', encoded[:, :, 1])
            # cv2.imwrite('/home/sthalham/gravity.png', encoded[:, :, 2])
            # cv2.imwrite('/home/sthalham/HHA.png', encoded)

            '''
            start_time = time.time()
            areaCol = encode_area(depth_inpaint)
            elapsed_time = time.time() - start_time
            print('encode area calculation time: ', elapsed_time)
            cv2.imwrite('/home/sthalham/visTests/clusters.png', areaCol)
            gravImg = geoCooFrame(normImg1, dptComp1)
            cv2.imwrite('/home/sthalham/visTests/gravArti.png', gravImg)

            print('testitest')
            
            
            edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)

            depth_im = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
            #rgb_im = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)

            #edges = edge_detection.detectEdges(np.float32(depth_im) / 255.0)
            edgesdep = edge_detection.detectEdges(np.float32(depth_im) / 255.0)
            orimapdep = edge_detection.computeOrientation(edgesdep)
            edgesdep = edge_detection.edgesNms(edgesdep, orimapdep)
            sca = 255.0 / np.amax(edgesdep)
            edgesdep = np.multiply(edgesdep, sca)
            edgesdep = edgesdep.astype(np.uint8)
            cv2.imwrite("/home/sthalham/strucedgedepth.png", edgesdep)

            
            edgesrgb = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
            orimaprgb = edge_detection.computeOrientation(edgesrgb)
            edgesrgb = edge_detection.edgesNms(edgesrgb, orimaprgb)

            edge_boxes = cv2.ximgproc.createEdgeBoxes()
            edge_boxes.setMaxBoxes(30)
            boxes = edge_boxes.getBoundingBoxes(edges, orimap)

            for b in boxes:
                x, y, w, h = b
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_AA)

            sca = 255.0 / np.amax(edgesrgb)
            edgesrgb = np.multiply(edgesrgb, sca)
            edgesrgb = edgesrgb.astype(np.uint8)
            cv2.imwrite("/home/sthalham/strucedgecompare.png", edgesrgb)

            '''
            print('stop')


    focalL = 1008.80026817 # blender focal length; just some line to place breakpoint
            # similar to t-less' focal length
            # focalL = 580.0  # according to microsoft
            # normImg = get_normal(depth_refine, fx=focalL, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5), for_vis=True)