#!/usr/bin/env python

import sys
import os
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage
import math
import datetime
import copy
import pickle

# mAP
# Precision = True positive / (True positive + False positive)
# Recall = True positive / (True positive + False negative)


def listDiff(first, second):
    # second = set(second)
    return [item for item in first if item not in second]


def boxoverlap(a, b):
    # Compute the symmetric intersection over union overlap between a set of
    # bounding boxes in a and a single bounding box in b.

    # a  a matrix where each row specifies a bounding box
    # b  a single bounding box

    # AUTORIGHTS
    # -------------------------------------------------------
    # Copyright (C) 2011-2012 Ross Girshick
    # Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick

    # This file is part of the voc-releaseX code
    # (http://people.cs.uchicago.edu/~rbg/latent/)
    # and is available under the terms of an MIT-like license
    # provided in COPYING. Please retain this notice and
    # COPYING if you use this file (or a portion of it) in
    # your project.
    # -------------------------------------------------------
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
    root = "/home/sthalham/workspace/EagleEye/json_results/"
    jsons = root + 'compFRCNN.json'
    #jsons = sys.argv[1]
    #dataset = sys.argv[2]
    dataset = 'tless'
    visu = True

    json_data = open(jsons).read()
    data = json.loads(json_data)


    testData = '/home/sthalham/data/LINEMOD/test/'
    #if dataset == 'tless':
    #    testData = '/home/sthalham/data/t-less_v2/test_primesense/'

    sub = os.listdir(testData)

    absObjs = 0

    if dataset is 'linemod':
        gtCatLst = [0] * 16
        detCatLst = [0] * 16
        falsePosLst = [0] * 16
        falseNegLst = [0] * 16
        allImg = 18273
    elif dataset is 'tless':
        gtCatLst = [0] * 31
        detCatLst = [0] * 31
        falsePosLst = [0] * 31
        falseNegLst = [0] * 31
        allImg = 7600

    proImg = 0

    for s in sub:

        #if s == '14' or s == '15' or s == '13' or s == '06' or s == '01' or s == '09' or s == '02' or s == '07' or s == '04' or s == '12' or s == '08' or s == '11':
        #    continue

        rgbPath = testData + s + "/rgb/"
        depPath = testData + s + "/depth/"
        gtPath = testData + s + "/gt.yml"

        with open(gtPath, 'r') as streamGT:
            gtYML = yaml.load(streamGT)

        subsub = os.listdir(rgbPath)

        counter = 0

        for ss in subsub:

            proImg = proImg + 1
            print('Processing image ', proImg, ' / ', allImg)

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
            for gt in gtImg:
                if dataset is 'linemod':
                    if gt['obj_id'] == s:
                        gtBoxes.append(gt['obj_bb'])
                        gtCats.append(gt['obj_id'])
                else:
                     gtCats.append(gt['obj_id'])
                     gtBoxes.append(gt['obj_bb'])

                #print(gt['obj_id'])
                #print(gt['obj_bb'])

            absObjs = absObjs + len(gtCats)  # increment all

            detBoxes = []
            detCats = []
            detSco = []
            for det in data:
                if det['image_id'] == img_id:
                    if dataset == 'linemod':

                        if det['category_id'] == s:
                            detBoxes.append(det['bbox'])
                            detCats.append(det['category_id'])
                            detSco.append(det['score'])
                    else:
                        detBoxes.append(det['bbox'])
                        detCats.append(det['category_id'])
                        detSco.append(det['score'])

                    #print(det['bbox'])
                    #print(det['category_id'])
                    #print(det['score'])


            if len(detBoxes) < 1:
                for i in gtCats:
                    gtCatLst[i] = gtCatLst[i] + 1
            else:

                #print('detBoxes: ', detBoxes)
                # legitimate, cause other objects are present but not annotated
                if dataset is 'linemod':
                    detBoxes = [detBoxes[detCats.index(s)]]
                    detSco = [detSco[detCats.index(s)]]
                    detCats = [detCats[detCats.index(s)]]

                    #print('detBoxes: ', detBoxes)
                    #print('detCats: ', detCats)

                    falsePos = []
                    truePos = []
                    for i, dC in enumerate(detCats):
                        for j, gC in enumerate(gtCats):
                            if dC is gC:
                                b1 = np.array([detBoxes[i][0], detBoxes[i][1], detBoxes[i][0] + detBoxes[i][2], detBoxes[i][1] + detBoxes[i][3]])
                                b2 = np.array([gtBoxes[j][0], gtBoxes[j][1], gtBoxes[j][0] + gtBoxes[j][2], gtBoxes[j][1] + gtBoxes[j][3]])
                                IoU = boxoverlap(b1, b2)
                                #print('IoU: ', IoU)
                                # occurences of 2 or more instances not possible in LINEMOD
                                if IoU > 0.5:
                                    truePos.append(dC)
                                else:
                                    falsePos.append(dC)
                            else:
                                if dataset is not 'linemod':
                                    falsePos.append(dC)

                    fp = falsePos
                    tp = truePos
                    fn = listDiff(gtCats, tp)

                else:
                    ind2rem = np.array([], dtype=np.uint8)

                    cleBoxes = []
                    cleSco = []
                    cleCats = []
                    for i, sco in enumerate(detSco):
                        if sco > 0.5:
                            cleBoxes.append(detBoxes[i])
                            cleSco.append(detSco[i])
                            cleCats.append(detCats[i])

                    detBoxes = cleBoxes
                    detSco = cleSco
                    detCats = cleCats


                    # find overlaping boxes
                    for i, iC in enumerate(detCats):
                        for j, jC in enumerate(detCats):
                            if i is not j:
                                b1 = np.array([detBoxes[i][0], detBoxes[i][1], detBoxes[i][0] + detBoxes[i][2],
                                               detBoxes[i][1] + detBoxes[i][3]])
                                b2 = np.array([detBoxes[j][0], detBoxes[j][1], detBoxes[j][0] + detBoxes[j][2],
                                               detBoxes[j][1] + detBoxes[j][3]])
                                IoU = boxoverlap(b1, b2)

                                # occurences of 2 or more instances not possible in LINEMOD
                                if IoU > 0.5:
                                    #print('IoU: ', IoU)
                                    #print(detSco[i], detSco[j])
                                    #print(iC, jC, i, j)
                                    if detSco[i] > detSco[j]:

                                        ind2rem = np.append(ind2rem, j)
                    # remove overlaping boxes
                    tempCats = detCats
                    ind2rem = np.unique(ind2rem)
                    ind2rem.sort()
                    for ind in reversed(ind2rem):
                        del detCats[ind]
                        del detBoxes[ind]

                    falsePos = []
                    truePos = []
                    for i, dC in enumerate(detCats):
                        for j, gC in enumerate(gtCats):
                            if dC is gC:
                                b1 = np.array([detBoxes[i][0], detBoxes[i][1], detBoxes[i][0] + detBoxes[i][2],
                                               detBoxes[i][1] + detBoxes[i][3]])
                                b2 = np.array([gtBoxes[j][0], gtBoxes[j][1], gtBoxes[j][0] + gtBoxes[j][2],
                                               gtBoxes[j][1] + gtBoxes[j][3]])
                                IoU = boxoverlap(b1, b2)
                                # print('IoU: ', IoU)
                                # occurences of 2 or more instances not possible in LINEMOD
                                if IoU > 0.5:
                                    truePos.append(dC)
                                else:
                                    falsePos.append(dC)
                            else:
                                if dataset is not 'linemod':
                                    falsePos.append(dC)

                    fp = falsePos
                    tp = truePos
                    fn = listDiff(gtCats, tp)

                # indexing with "target category" only possible due to linemod annotation
                if dataset == 'linemod':

                    gtCatLst[s] = gtCatLst[s] + 1
                    detCatLst[s] = detCatLst[s] + len(tp)
                    falsePosLst[s] = falsePosLst[s] + len(fp)
                    falseNegLst[s] = falseNegLst[s] + 1

                else:
                    for i, gt in enumerate(gtCats):
                        gtCatLst[gt] = gtCatLst[gt] + 1
                    for i, pos in enumerate(tp):
                        detCatLst[pos] = detCatLst[pos] +1
                    for i, neg in enumerate(fp):
                        falsePosLst[neg] = falsePosLst[neg] + 1
                    for i, fneg in enumerate(fn):
                        falseNegLst[fneg] = falseNegLst[fneg] + 1

                # VISUALIZATION
                if visu == True:
                    img = cv2.imread(rgbImgPath, -1)
                    for i, bb in enumerate(detBoxes):
                        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                                          (0, 0, 0), 3)
                        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                                          (250, 215, 10), 2)

                    for i, bb in enumerate(detBoxes):
                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (int(bb[0])+5, int(bb[1])+int(bb[3])-5)
                        fontScale = 0.5
                        fontColor = (25, 215, 250)
                        fontthickness = 2
                        lineType = 2
                        if dataset:
                            if detCats[i] == 1: cate = 'ape'
                            elif detCats[i] == 2: cate = 'benchvise'
                            elif detCats[i] == 3: cate = 'bowl'
                            elif detCats[i] == 4: cate = 'camera'
                            elif detCats[i] == 5: cate = 'can'
                            elif detCats[i] == 6: cate = 'cat'
                            elif detCats[i] == 7: cate = 'cup'
                            elif detCats[i] == 8: cate = 'drill'
                            elif detCats[i] == 9: cate = 'duck'
                            elif detCats[i] == 10: cate = 'eggbox'
                            elif detCats[i] == 11: cate = 'glue'
                            elif detCats[i] == 12: cate = 'holepuncher'
                            elif detCats[i] == 13: cate = 'iron'
                            elif detCats[i] == 14: cate = 'lamp'
                            elif detCats[i] == 15: cate = 'phone'
                            gtText = cate
                        else:
                            gtText = str(detCats[i])
                        #gtText = cate + " / " + str(detSco[i])

                        fontColor2 = (0, 0, 0)
                        fontthickness2 = 3
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

                    cv2.imwrite('/home/sthalham/visTests/detectRetNet.jpg', img)

                    print('STOP')


        #detAcc = detCatLst[s] / gtCatLst[s]
        #print('accuracy category ', s, ': ', detAcc)

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

    #print('STOP')


    # Precision = True positive / (True positive + False positive)
    # Recall = True positive / (True positive + False negative)
    if dataset is 'linemod':
        detAcc = [0] * 16
        detPre = [0] * 16
        detRec = [0] * 16
    elif dataset is 'tless':
        detAcc = [0] * 31
        detPre = [0] * 31
        detRec = [0] * 31

    for ind, cat in enumerate(gtCatLst):
        if ind == 0:
            continue
        detAcc[ind] = detCatLst[ind] / cat
        print('accuracy category ', ind, ': ', detAcc[ind])

        if (detCatLst[ind] + falsePosLst[ind]) == 0:
            detPre[ind] = 0.0
        else:
            detPre[ind] = detCatLst[ind] / (detCatLst[ind] + falsePosLst[ind])
        if (detCatLst[ind] + falseNegLst[ind]) == 0:
            detRec[ind] = 0.0
        else:
            detRec[ind] = detCatLst[ind] / (detCatLst[ind] + falseNegLst[ind])

        #print('precision category ', ind, ': ', detPre[ind])
        #print('recall category ', ind, ': ', detRec[ind])

    print('accuracy overall: ', sum(detAcc)/len(detAcc))
    #print('mAP: ', sum(detPre) / len(detPre))
    #print('mAR: ', sum(detRec) / len(detRec))


