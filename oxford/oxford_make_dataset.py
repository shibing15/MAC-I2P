import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms
import bisect

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from oxford import augmentation
from oxford import options
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
import multiprocessing
from multiprocessing import Pool

def read_train_val_split(txt_path):
    with open(txt_path) as f:
        sets = [x.rstrip() for x in f.readlines()]
    traversal_list = list(sets)
    return traversal_list

def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale


def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop

def downsample_with_reflectance(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance



def downsample_np_(pc_np, intensity_np, k):
    if pc_np.shape[1] >= k:
        choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
    else:
        fix_idx = np.asarray(range(pc_np.shape[1]))
        while pc_np.shape[1] + fix_idx.shape[0] < k:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
        random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
    pc_np = pc_np[:, choice_idx]
    intensity_np = intensity_np[:, choice_idx]

    return pc_np, intensity_np


def clear_traversal_novoerlap_dat(opt, traversal,mode,
                                pc_idx,cam_t_idx,
                                pc_timestamps_list,
                                camera_timestamps_list,
                                P_cam_pc):
        flag = 0 
        K = np.asarray([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]], dtype=np.float32)
        # print('find camera time success', camera_timestamp_idx)
        camera_folder = os.path.join(opt.dataroot, traversal, 'stereo', 'centre')
        camera_timestamp = camera_timestamps_list[cam_t_idx]
        # print(os.path.join(camera_folder, "%d.jpg" % camera_timestamp))
        img = np.array(Image.open(os.path.join(camera_folder, "%d.jpg" % camera_timestamp)))
        # print(traversal, pc_timestamp, pc_timestamp_idx, camera_folder,camera_timestamp ,'load img success')
        # ------------- load image, original size is 960x1280, bottom rows are car itself -------------
        tmp_img_H = img.shape[0]
        img = img[0:(tmp_img_H-opt.crop_original_bottom_rows), :, :]
        # scale
        img = cv2.resize(img,
                         (int(round(img.shape[1] * opt.img_scale)), int(round((img.shape[0] * opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
        # img = Image.fromarray(img)
        K = camera_matrix_scaling(K, opt.img_scale)
        # random crop into input size
        if 'train' == mode:
            img_crop_dx = random.randint(0, img.shape[1] - opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + opt.img_H,
              img_crop_dx:img_crop_dx + opt.img_W, :]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        K_4 = camera_matrix_scaling(K, 0.25)

        # ------------- load point cloud ----------------
        if opt.is_remove_ground:
            lidar_name = 'lms_front_foreground'
        else:
            lidar_name = 'lms_front'
        pc_path = os.path.join(opt.dataroot, traversal, lidar_name, '%d.npy' % pc_timestamps_list[pc_idx])
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        # print(traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num, 'load pc success')
        # limit max_z, the pc is in CAMERA coordinate
        pc_np_x_square = np.square(pc_np[0, :])
        pc_np_z_square = np.square(pc_np[2, :])
        pc_np_range_square = pc_np_x_square + pc_np_z_square
        pc_mask_range = pc_np_range_square < opt.pc_max_range * opt.pc_max_range
        pc_np = pc_np[:, pc_mask_range]
        intensity_np = intensity_np[:, pc_mask_range]
        
        # remove the ground points!

        if pc_np.shape[1] > 2 * opt.input_pt_num:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np = downsample_with_reflectance(pc_np, intensity_np[0], voxel_grid_downsample_size=0.2)
            intensity_np = np.expand_dims(intensity_np, axis=0)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
        # random sampling
        pc_np, intensity_np = downsample_np_(pc_np, intensity_np, opt.input_pt_num)
        
        #get pc_mask 
        pc_cam = np.dot(P_cam_pc[0:3, 0:3], pc_np) + P_cam_pc[0:3, 3:]
        pc_ = np.dot(K_4, pc_cam)
        depth = pc_[2, :]
        if(np.sum(depth == 0) > 0):
            return flag
        pc_mask = np.zeros((1, np.shape(pc_np)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2, :]
        xy = np.floor(pc_[0:2, :])
        is_in_picture = (depth > 0) & (xy[0, :] >= 0) & (xy[0, :] <= (opt.img_W * 0.25 - 1)) & (xy[1, :] >= 0) & (
                xy[1, :] <= (opt.img_H * 0.25 - 1)) & (pc_[2, :] > 0)
        pc_mask[:, is_in_picture] = 1.
        if(np.sum(pc_mask) > 6000):
            flag = 1
        return flag


def clear_traversal_data(opt, mode,
                        pc_timestamps_list,
                        pc_poses_np,
                        img_timestamps_list,
                        img_poses_np,
                        traversal_pc_num,
                        traversal):
        '''
        clear data for  traversal
        
        '''
        #init output result
        new_pc_timestamps_list = []
        new_pc_poses_list = []
        new_camera_timestamp_list = []
        new_camera_poses_list = []
        #selcet mode and chose translation limitions
        if mode == 'train':
            translation_max = opt.translation_max
        else:
            translation_max = opt.test_translation_max
        # pc is built every opt.pc_build_interval (2m),
        # so search for the previous/nex pc_timestamp that > max_translation
        #对每段数据的每个点云进行数据清洗
        for idx in range(len(pc_timestamps_list)):
            
            #根据点云时间戳确定对应图像的时间戳范围
            index_interval = math.ceil(translation_max / opt.pc_build_interval)
            previous_pc_t_idx = max(0, idx - index_interval)
            previous_pc_t = pc_timestamps_list[previous_pc_t_idx]
            next_pc_t_idx = min(traversal_pc_num - 1, idx + 5)
            next_pc_t = pc_timestamps_list[next_pc_t_idx]
            P_o_pc = pc_poses_np[idx]

            previous_camera_t_idx = bisect.bisect_left(img_timestamps_list, previous_pc_t)
            next_camer_t_idx = bisect.bisect_left(img_timestamps_list, next_pc_t)

            #在对应图像范围内寻找重合度和位移均符合标准的图像
            for cam_t_idx in range(previous_camera_t_idx,next_camer_t_idx+1):
                cam_t_idx = random.randint(previous_camera_t_idx, next_camer_t_idx)
                if cam_t_idx >= img_poses_np.shape[0]:
                    cam_t_idx = img_poses_np.shape[0] - 1
                P_o_cam = img_poses_np[cam_t_idx]
                #图像和额点云的相对位姿
                P_cam_pc = np.dot(np.linalg.inv(P_o_cam), P_o_pc)
                #计算图像与点云的相对位移
                t_norm = np.linalg.norm(P_cam_pc[0:3, 3])
                #判定该图像是否与点云有足够的重叠区域
                voerlap_flag = clear_traversal_novoerlap_dat(opt, traversal, mode,
                                                            idx, cam_t_idx, 
                                                            pc_timestamps_list,
                                                            img_timestamps_list,
                                                            P_cam_pc)
                if (t_norm < 10) & (voerlap_flag == 1):
                    # print('%s pointcloud %s and img %s have been selected'%(traversal, pc_timestamps_list[idx], img_timestamps_list[cam_t_idx]))
                    new_pc_timestamps_list.append(pc_timestamps_list[idx])
                    new_pc_poses_list.append(pc_poses_np[idx])
                    new_camera_timestamp_list.append(img_timestamps_list[cam_t_idx])
                    new_camera_poses_list.append(img_poses_np[cam_t_idx])
                    break
            
        return  new_pc_timestamps_list, np.array(new_pc_poses_list), \
                new_camera_timestamp_list, np.array(new_camera_poses_list)

def clean_data_for_traversal(root_path, traversal, mode):
           #data load
        P_convert = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        P_convert_inv = np.linalg.inv(P_convert)
        pc_timestamps_np = np.load(os.path.join(root_path, traversal, 'pc_timestamps.npy')).tolist()
        pc_poses_np = np.load(os.path.join(root_path, traversal, 'pc_poses.npy')).astype(np.float32)
        # convert it to camera coordinate，将点云转换至图像坐标系
        for b in range(pc_poses_np.shape[0]):
            pc_poses_np[b] = np.dot(P_convert, np.dot(pc_poses_np[b], P_convert_inv))
        img_timestamps_np = np.load(os.path.join(root_path, traversal, 'camera_timestamps.npy')).tolist()
        img_poses_np = np.load(os.path.join(root_path, traversal, 'camera_poses.npy')).astype(np.float32)
        # convert it to camera coordinate,图像pose转换至图像坐标系
        for b in range(img_poses_np.shape[0]):
            img_poses_np[b] = np.dot(P_convert, np.dot(img_poses_np[b], P_convert_inv))

        #data clear 
        new_pc_timestamps_list, new_pc_poses_np, \
        new_camera_timestamp_list, new_camera_poses_np = clear_traversal_data(opt, mode,
                                                                            pc_timestamps_np,
                                                                            pc_poses_np,
                                                                            img_timestamps_np,
                                                                            img_poses_np,
                                                                            len(pc_timestamps_np),
                                                                            traversal)

        new_pc_timestamps_list_np = np.array(new_pc_timestamps_list)
        new_camera_timestamp_list_np = np.array(new_camera_timestamp_list)
        np.save(os.path.join(root_path, traversal, 'pc_timestamps_siami2p.npy'), new_pc_timestamps_list_np)
        np.save(os.path.join(root_path, traversal, 'pc_poses_siami2p.npy'), new_pc_poses_np)
        np.save(os.path.join(root_path, traversal, 'camera_timestamps_siami2p.npy'), new_camera_timestamp_list_np)
        np.save(os.path.join(root_path, traversal, 'camera_poses_siami2p.npy'), new_camera_poses_np)
        print('%s \n %s \n %s \n %s \n has been save'
            %(os.path.join(root_path, traversal, 'pc_timestamps_siami2p.npy'),
            os.path.join(root_path, traversal, 'pc_poses_siami2p.npy'),
            os.path.join(root_path, traversal, 'camera_timestamps_siami2p.npy'),
            os.path.join(root_path, traversal, 'camera_poses_siami2p.npy')))

def make_newoxford_dataset(root_path, mode, opt):
    #init output pc and camera data and pose list,dict
    dataset = []
    pc_timestamps_list_dict = {}
    pc_poses_np_dict = {}
    camera_timestamps_list_dict = {}
    camera_poses_np_dict = {}
    #相机内参数
    K = np.asarray([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]], dtype=np.float32)

    #selcet mode
    if mode == 'train':
        seq_list = read_train_val_split(os.path.join(root_path, 'train.txt'))
    elif 'val' in mode:
        seq_list = read_train_val_split(os.path.join(root_path, 'val.txt'))
    else:
        raise Exception('Invalid mode.')

    #多线程参数
    thread_pool = Pool(5)
    threads = []
    #对每段数据进行清洗
    for traversal in seq_list:
        print(traversal, ' start data clean and selcet')
        thread_pool.apply_async(clean_data_for_traversal, args=((root_path, traversal, mode)))
    print('等待所有线程完成')
    thread_pool.close()
    thread_pool.join()

        

    return dataset, \
           pc_timestamps_list_dict, pc_poses_np_dict, \
           camera_timestamps_list_dict, camera_poses_np_dict





if __name__=='__main__':
    opt = options.Options()
    
    dataset, \
    pc_timestamps_list_dict, pc_poses_np_dict, \
    camera_timestamps_list_dict, camera_poses_np_dict =make_newoxford_dataset(opt.dataroot, 'val', opt)
