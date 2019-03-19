import os,sys
import bpy
import numpy as np
import transforms3d as tf3d
from random import randint
from random import random
from random import gauss
from random import uniform
import cv2
import yaml
import itertools
from math import radians,degrees,tan,cos
from numpy.linalg import inv
import OpenEXR, Imath

import bmesh
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

print("Starting script")

base_dir = "/home/sthalham/data/LINEMOD/models_stl"
gt_dir = "/home/sthalham/data/renderings/linemod_BG"
sample_dir = '/home/sthalham/data/renderings/linemod_crops' #directory for temporary files (cam_L, cam_R, masks..~)
target_dir = '/home/sthalham/data/renderings/linemod_crops/patches'
depPath = gt_dir + "/depth/"
partPath = gt_dir + "/part/"
gtPath = gt_dir
maskPath = gt_dir + "/mask/"

index=0

if not(os.path.exists(target_dir+"/depth")):
    os.makedirs(target_dir+"/depth")

model_file=[]
model_solo=[]
for root, dirs, files in os.walk(base_dir):
    for file in sorted(files):
        if file.endswith(".stl"):
             temp_fn =os.path.join(root, file)
             model_file.append(temp_fn)
             model_solo.append(file)
             #print(len(model_file),temp_fn)

print(gtPath)
counter = 0

for gt in os.listdir(gtPath):
    if not gt.endswith(".yaml"):
        continue
    counter += 1
    
    gtfile = gtPath +'/' + gt

    with open(gtfile, 'r') as stream:
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
    
    object_label =[]
    anchor_pose = np.zeros((6))
    print(anchor_pose)
            
    for i, bb in enumerate(bboxes[:-1]):
    
        bpy.ops.object.select_all(action='DESELECT')
        scene = bpy.context.scene
        scene.objects.active = bpy.data.objects["template"]
        for obj in scene.objects:
            if obj.type == 'MESH':
                if obj.name == 'template':
                    obj.select = False          
                elif obj.name == 'InvisibleCube':
                    obj.select = False
                else:
                    obj.select = True

        bpy.ops.object.delete()
        bpy.ops.object.select_all(action='DESELECT')
        obj_object = bpy.data.objects["template"]
        obj_object.pass_index = 1
        mat = obj_object.active_material
        
        file_model = model_file[bboxes[i][0]]
        solo_model = model_solo[bboxes[i][0]]
        imported_object = bpy.ops.import_mesh.stl(filepath=file_model, filter_glob="*.stl", files=[{"name":solo_model, "name":solo_model}], directory=root)
        object_label.append(bboxes[0][0])
        obj_object = bpy.context.selected_objects[0]
        obj_object.active_material = mat
        obj_object.pass_index = i +2 # don't add?
        anchor_pose[0] = poses[i, 4]
        anchor_pose[1] = poses[i, 5]
        anchor_pose[2] = poses[i, 6]
        rot = tf3d.euler.quat2euler(poses[i, 0:4])
        anchor_pose[3] = rot[0]
        anchor_pose[4] = rot[1]
        anchor_pose[5] = rot[2]
    
        #bpy.ops.rigidbody.object_settings_copy()
        scene.frame_set(0)
        for obj in scene.objects:
            if obj.type == 'MESH':
                obj_object= bpy.data.objects[obj.name]
            
            if obj_object.pass_index>1:
                idx = obj_object.pass_index -2
                obj_object.location.x=anchor_pose[0]
                obj_object.location.y=anchor_pose[1]
                obj_object.location.z=anchor_pose[2]
                obj_object.rotation_euler.x= anchor_pose[3]
                obj_object.rotation_euler.y= anchor_pose[4]
                obj_object.rotation_euler.z= anchor_pose[5]
                 
                # assign different color
                rand_color = (random(), random(), random())
                obj_object.active_material.diffuse_color = rand_color
                if obj_object.pass_index > 1:
                    obj_object.pass_index = 0
                
        tree = bpy.context.scene.node_tree
        nodes = tree.nodes

        crop_name = gt[:-8] + '_' + str(bboxes[i][0]) + '_depth.exr'
        depthfile = os.path.join(target_dir+'/depth', crop_name)

        for ob in scene.objects:
            if ob.type == 'CAMERA':          
                if ob.name=='cam_L': #ob.name =='mask':
                    #Render IR image and Mask
                    bpy.context.scene.camera = ob
                    print('Set camera %s for IR' % ob.name )
                    file_L = os.path.join(sample_dir , ob.name )
                    
                    auto_file_depth = os.path.join(sample_dir+'/temp/', ob.name+'0000.exr')
                    node= nodes['depthout']
                    node.file_slots[0].path = ob.name
                    node_mix = nodes['Render Layers']
                    link_depth = tree.links.new(node_mix.outputs["Z"], node.inputs[0])
                    node.base_path=sample_dir+'/temp/'
                  
                    scene.render.filepath = file_L
                    bpy.ops.render.render( write_still=True )
                    tree.links.remove(link_depth)
                  
                    os.rename(auto_file_depth, depthfile)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        golden = OpenEXR.InputFile(depthfile)
        dw = golden.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        redstr = golden.channel('R', pt)
        depth = np.fromstring(redstr, dtype=np.float32)
        depth.shape = (size[1], size[0])

        centerX = depth.shape[1] / 2.0
        centerY = depth.shape[0] / 2.0
        fov = 57.8

        uv_table = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.int16)
        column = np.arange(0, depth.shape[0])
        uv_table[:, :, 1] = np.arange(0, depth.shape[1]) - centerX
        uv_table[:, :, 0] = column[:, np.newaxis] - centerY
        uv_table = np.abs(uv_table)

        depth = depth * np.cos(np.radians(fov / depth.shape[1] * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / depth.shape[1] * uv_table[:, :, 0]))
    
        depth[depth == np.inf] = 0
    