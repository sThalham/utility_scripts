import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import time
import itertools
#import pcl
#from pcl import pcl_visualization

import OpenEXR, Imath
from pathlib import Path

#kin_res_x = 640
#kin_res_y = 480
resX = 640
resY = 480
# fov = 1.0088002681732178
fov = 57.8
fxkin = 579.68  # blender calculated
fykin = 542.31  # blender calculated
cxkin = 320
cykin = 240
depthCut = 1800.0

np.set_printoptions(threshold=np.nan)


def encodeImage(depth):
    img = np.zeros((resY, resX, 3), dtype=np.uint8)

    normImg, depImg = get_normal(depth, fxkin, fykin, cxkin, cykin, for_vis=True)
    img[:, :, 0] = compute_disparity(depImg)
    img[:, :, 1] = encode_area(depImg)
    img[:, :, 2] = compute_angle2gravity(normImg, depImg)

    return img


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


def encode_area(depth):

    onethird = cv2.resize(depth, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    pc = create_point_cloud(onethird, fxkin, fykin, cxkin, cykin, 1.0)
    cloud = pcl.PointCloud(np.array(pc, dtype=np.float32))
    CPPInput = "/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/tempCloud.pcd"
    CPPOutput = "/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/output.pcd"
    pcl.save(cloud, CPPInput)

    args = ("/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/conditional_euclidean_clustering", CPPInput, CPPOutput)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    cloudNew = pcl.load_XYZI(CPPOutput)
    pcColor = cloudNew.to_array()
    inten = pcColor[:, 3]
    inten = np.reshape(inten, (int(resY/3), int(resX/3)))

    clusters, surf = np.unique(inten, return_counts=True)

    flat = surf.flatten()
    flat.sort()
    area_ref = np.mean(flat[0:-1])
    area_max = np.nanmax(surf)
    areaCol = np.ones(inten.shape, dtype=np.uint8)

    for i, cl in enumerate(clusters):
        if surf[i] < area_ref:
            mask = np.where(inten == cl, True, False)
            val = 255.0 - ((surf[i] / area_ref) * 127.5)  # prob.: 255 - ...
            val = val.astype(dtype=np.uint8)
            areaCol = np.where(mask, val, areaCol)

        else:
            mask = np.where(inten == cl, True, False)
            val = 127.5 - ((surf[i] / area_max) * 126.5)  # prob.: 255 - ...
            val = val.astype(dtype=np.uint8)
            areaCol = np.where(mask, val, areaCol)

    areaCol = cv2.resize(areaCol, (resX, resY), interpolation=cv2.INTER_NEAREST)

    areaCol = np.where(depth > depthCut, 0, areaCol)

    return areaCol


def compute_disparity(depth):
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

    return disp_final


def compute_angle2gravity(normals, depth):
    r, c, p = normals.shape
    mask = depth < depthCut
    normals[:, :, 0] = np.where(mask, normals[:, :, 0], np.NaN)
    normals[:, :, 1] = np.where(mask, normals[:, :, 1], np.NaN)
    normals[:, :, 2] = np.where(mask, normals[:, :, 2], np.NaN)

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

    return gImg


def manipulate_depth(fn_gt, fn_depth, fn_part):

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        bboxes = np.zeros((len(query), 5), np.int)
        poses = np.zeros((len(query), 7), np.float32)
        mask_ids = np.zeros((len(query)), np.int)
        for j in range(len(query)-1): # skip cam pose
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

    if bboxes.shape[0] < 2:
        print('invalid train image, no bboxes in fov')
        return None, None, None, None

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(fn_depth)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depth = np.fromstring(redstr, dtype=np.float32)
    depth.shape = (size[1], size[0])

    centerX = depth.shape[1] / 2.0
    centerY = depth.shape[0] / 2.0

    uv_table = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.int16)
    column = np.arange(0, depth.shape[0])
    uv_table[:, :, 1] = np.arange(0, depth.shape[1]) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / depth.shape[1] * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / depth.shape[1] * uv_table[:, :, 0]))

    if np.nanmean(depth) < 0.5 or np.nanmean(depth) > 2.0:
        print('invalid train image; range is wrong')
        return None, None, None, None

    depthFinal = cv2.resize(depth, (resX, resY))

    return depth, bboxes, poses, mask_ids


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, for_vis=True):
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
    uv_table = np.abs(uv_table)

    # kernel = np.ones((5, 5), np.uint8)
    # depth_refine = cv2.dilate(depth_refine, kernel, iterations=1)
    # depth_refine = cv2.medianBlur(depth_refine, 5 )
    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)
    # depth_refine = ndimage.uniform_filter(depth_refine, size=11)

    # very_blurred = ndimage.gaussian_filter(face, sigma=5)
    v_x = np.zeros((res_y, res_x, 3))
    v_y = np.zeros((res_y, res_x, 3))
    normals = np.zeros((res_y, res_x, 3))

    dig = np.gradient(depth_refine, 2, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant * dig[0]
    v_y[:, :, 1] = depth_refine * constant + (uv_table_sign[:, :, 0] * constant) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant + uv_table_sign[:, :, 1] * constant * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
    # norm[norm == 0] = 1

    cross = cross / norm
    cross = cross.reshape(res_y, res_x, 3)
    cross = np.abs(cross)
    cross = np.nan_to_num(cross)

    # cross_ref = np.copy(cross)
    # cross[cross_ref==[0,0,0]]=0 #set zero for nan values

    # cam_angle = np.arccos(cross[:, :, 2])
    # cross[np.abs(cam_angle) > math.radians(75)] = 0  # high normal cut
    #cross[depth_refine <= 200] = 0  # 0 and near range cut
    cross[depth_refine > depthCut] = 0  # far range cut
    if not for_vis:
        scaDep = 1.0 / np.nanmax(depth_refine)
        depth_refine = np.multiply(depth_refine, scaDep)
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))
        scaCro = 255.0 / np.nanmax(cross)
        cross = np.multiply(cross, scaCro)
        cross = cross.astype(np.uint8)

    return cross, depth_refine


##########################
#         MAIN           #
##########################
if __name__ == "__main__":

    root = '/home/sthalham/data/t-less_v2/test_kinect/'
    target = '/home/sthalham/data/prepro/tl_baseline/'

    sub = os.listdir(root)

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

    dictVal = copy.deepcopy(dict)

    annoID = 0

    gloCo = 0
    allCo = 49000
    times = []

    for s in sub:
        rgbPath = root + s + "/rgb/"
        depPath = root + s + "/depth/"
        gtPath = root + s + "/gt.yml"
        infoPath = root + s + "/info.yml"

        with open(infoPath, 'r') as stream:
            opYML = yaml.load(stream)

        with open(gtPath, 'r') as streamGT:
            gtYML = yaml.load(streamGT)

        subsub = os.listdir(rgbPath)

        counter = 0
        for ss in subsub:
            print(s)

            start_time = time.time()
            gloCo = gloCo + 1

            imgname = ss
            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss

            # print(rgbImgPath)

            if ss.startswith('000'):
                ss = ss[3:]
            elif ss.startswith('00'):
                ss = ss[2:]
            elif ss.startswith('0'):
                ss = ss[1:]
            ss = ss[:-4]

            # calib = opYML[int(ss)]
            # K = calib["cam_K"]
            # depSca = calib["depth_scale"]
            # fxkin = K[0]
            # fykin = K[4]
            # cxx = K[2]
            # cyy = K[5]
            # cam_R = calib["cam_R_w2c"]
            # cam_T = calib["cam_t_w2c"]

            #########################
            # Prepare the stuff
            #########################

            # read images and create mask
            rgbImg = cv2.imread(rgbImgPath)
            depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
            rows, cols = depImg.shape
            depImg = np.multiply(depImg, 0.1)

            # inpainting
            #scaleOri = np.amax(depImg)
            #inPaiMa = np.where(depImg == 0.0, 255, 0)
            #inPaiMa = inPaiMa.astype(np.uint8)
            #inPaiDia = 5.0
            #depth_refine = depImg.astype(np.float32)
            #depPaint = cv2.inpaint(depth_refine, inPaiMa, inPaiDia, cv2.INPAINT_NS)

            #depNorm = depPaint - np.amin(depPaint)
            #rangeD = np.amax(depNorm)
            #depNorm = np.divide(depNorm, rangeD)
            #depth_refine = np.multiply(depNorm, scaleOri)

            drawN = [1, 1, 1, 2]
            freq = np.bincount(drawN)
            rnd = np.random.choice(np.arange(len(freq)), 1, p=freq / len(drawN), replace=False)

            # change drawN if you want a data split
            # print("storage choice: ", rnd)
            #rnd = 1
            #if s != '02':
            if rnd == 2 and len(os.listdir(target + 'coco_train2014')) < 10000:

                # create image number and name
                template = '00000'
                s = int(s)
                ssm = int(ss) + 1
                pre = (s - 1) * 1296
                img_id = pre + ssm
                tempSS = template[:-len(str(img_id))]

                imgNum = str(img_id)
                imgNam = tempSS + imgNum + '.jpg'
                iname = str(imgNam)

                fileName = target + '/coco_train2014/' + imgNam
                myFile = Path(fileName)
                if myFile.exists():
                    print('File exists, skip encoding')
                else:
                    # imgI = encodeImage(depImg)
                    imgI, depth_refine = get_normal(depImg, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin, for_vis=False)
                    cv2.imwrite(fileName, imgI)

                gtImg = gtYML[int(ss)]

                # bbsca = 320.0 / 640.0
                for i in range(len(gtImg)):
                    curlist = gtImg[i]
                    obj_id = curlist["obj_id"]
                    obj_bb = curlist["obj_bb"]
                    # for ind, ob in enumerate(obj_bb):
                    #    obj_bb[ind] = int(ob * bbsca)

                    area = obj_bb[2] * obj_bb[3]

                    nx1 = obj_bb[0]
                    ny1 = obj_bb[1]
                    nx2 = nx1 + obj_bb[2]
                    ny2 = ny1 + obj_bb[3]
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    cont = npseg.tolist()

                    annoID = annoID + 1
                    tempVa = {
                        "id": annoID,
                        "image_id": img_id,
                        "category_id": obj_id,
                        "bbox": obj_bb,
                        "segmentation": [cont],
                        "area": area,
                        "iscrowd": 0
                    }
                    dictVal["annotations"].append(tempVa)

                # create dictionaries for json
                tempVl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": img_id,
                    "name": iname
                }
                dictVal["licenses"].append(tempVl)

                tempVi = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": iname,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": img_id
                }
                dictVal["images"].append(tempVi)

            else:
                # create image number and name
                template = '00000'
                s = int(s)
                si = s
                ssm = int(ss) + 1
                pre = (s - 1) * 1296
                img_id = pre + ssm
                tempSS = template[:-len(str(img_id))]

                imgNum = str(img_id)
                imgNam = tempSS + imgNum + '.jpg'
                iname = str(imgNam)

                fileName = target + '/coco_val2014/' + imgNam
                myFile = Path(fileName)
                if myFile.exists():
                    print('File exists, skip encoding')
                else:
                    # imgI = encodeImage(depImg)
                    imgI, depth_refine = get_normal(depImg, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin, for_vis=False)
                    cv2.imwrite(fileName, imgI)

                gtImg = gtYML[int(ss)]

                detBoxes = []
                detGT = []
                for i in range(len(gtImg)):
                    curlist = gtImg[i]
                    obj_id = curlist["obj_id"]
                    obj_bb = curlist["obj_bb"]

                    obid = obj_id
                    if obid < 10:
                        obid = '_0' + str(obid) + '_r.npy'
                    else:
                        obid = '_' + str(obid) + '_r.npy'

                    checkR = root[:-6] + '/mask/' + imgname[:-4] + obid
                    #visibility = np.load(checkR)
                    #print(obj_id, " visibility: ", visibility, "; ", "BBOX: ", obj_bb)
                    #if visibility < 0.5:
                    #    print('too low objects visibility')
                    #    continue

                    area = obj_bb[2] * obj_bb[3]

                    nx1 = obj_bb[0]
                    ny1 = obj_bb[1]
                    nx2 = nx1 + obj_bb[2]
                    ny2 = ny1 + obj_bb[3]
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    cont = npseg.tolist()

                    bb = (nx1, ny1, nx2, ny2)
                    detBoxes.append(bb)
                    detGT.append(obj_id)

                    annoID = annoID + 1
                    tempVa = {
                        "id": annoID,
                        "image_id": img_id,
                        "category_id": obj_id,
                        "bbox": obj_bb,
                        "segmentation": [cont],
                        "area": area,
                        "iscrowd": 0
                    }
                    dict["annotations"].append(tempVa)

                # create dictionaries for json
                tempVl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": img_id,
                    "name": iname
                }
                dict["licenses"].append(tempVl)

                tempVi = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": iname,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": img_id
                }
                dict["images"].append(tempVi)

                #for i, bb in enumerate(detBoxes):
                #    cv2.rectangle(rgbImg, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                #                  (255, 255, 255), 3)
                #    cv2.rectangle(rgbImg, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                #                  (0, 0, 0), 1)

                #    font = cv2.FONT_HERSHEY_SIMPLEX
                #    bottomLeftCornerOfText = (int(bb[0]), int(bb[1]))
                #    fontScale = 1
                #    fontColor = (0, 0, 0)
                #    fontthickness = 1
                #    lineType = 2
                #    gtText = str(detGT[i])

                #    fontColor2 = (255, 255, 255)
                #    fontthickness2 = 3
                #    cv2.putText(rgbImg, gtText,
                #                bottomLeftCornerOfText,
                #                font,
                #                fontScale,
                #                fontColor2,
                #                fontthickness2,
                #                lineType)

                #    cv2.putText(rgbImg, gtText,
                #                bottomLeftCornerOfText,
                #                font,
                #                fontScale,
                #                fontColor,
                #                fontthickness,
                #                lineType)

                #cv2.imwrite('/home/sthalham/visTests/detect.jpg', rgbImg)

                #print('STOP')

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times) / len(times)
            eta = ((allCo - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')
                times = []

    catsInt = range(1, 31)

    for s in catsInt:
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)
        dictVal["categories"].append(tempC)

    trainAnno = target + "/annotations/instances_val2014.json"
    valAnno = target + "/annotations/instances_train2014.json"

    with open(valAnno, 'w') as fpV:
        json.dump(dictVal, fpV)

    with open(trainAnno, 'w') as fpT:
       json.dump(dict, fpT)

    print('everythings done')
