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
np.seterr(divide='ignore', invalid='ignore')

def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale


def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int32)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i + 1], pts))
        return farthest_pts, farthest_pts_idx


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


def read_train_val_split(txt_path):
    with open(txt_path) as f:
        sets = [x.rstrip() for x in f.readlines()]
    traversal_list = list(sets)
    return traversal_list


def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))


def make_oxford_dataset(root_path, mode, opt):
    #init output pc and camera data and pose list,dict
    dataset = []
    pc_timestamps_list_dict = {}
    pc_poses_np_dict = {}
    camera_timestamps_list_dict = {}
    camera_poses_np_dict = {}

    #selcet mode
    if mode == 'train':
        seq_list = read_train_val_split(os.path.join(root_path, 'train.txt'))
    elif 'val' in mode:
        seq_list = read_train_val_split(os.path.join(root_path, 'val.txt'))
    else:
        raise Exception('Invalid mode.')

    for traversal in seq_list:
        #data load
        pc_timestamps_np = np.load(os.path.join(root_path, traversal, 'pc_timestamps_siami2p.npy')).tolist()
        pc_poses_np = np.load(os.path.join(root_path, traversal, 'pc_poses_siami2p.npy')).astype(np.float32)
        img_timestamps_np = np.load(os.path.join(root_path, traversal, 'camera_timestamps_siami2p.npy')).tolist()
        img_poses_np = np.load(os.path.join(root_path, traversal, 'camera_poses_siami2p.npy')).astype(np.float32)
        
        #make data
        pc_timestamps_list_dict[traversal] = pc_timestamps_np
        pc_poses_np_dict[traversal] = pc_poses_np
        camera_timestamps_list_dict[traversal] = img_timestamps_np
        camera_poses_np_dict[traversal] = img_poses_np

        for i in range(len(pc_timestamps_np)):
            pc_timestamp = pc_timestamps_np[i]
            camera_timestamp = img_timestamps_np[i]
            dataset.append((traversal, pc_timestamp, camera_timestamp, i, len(pc_timestamps_np), len(img_timestamps_np)))

    return dataset, \
           pc_timestamps_list_dict, pc_poses_np_dict, \
           camera_timestamps_list_dict, camera_poses_np_dict

class OxfordLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(OxfordLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # list of (traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num)
        self.dataset, \
        self.pc_timestamps_list_dict, self.pc_poses_np_dict, \
        self.camera_timestamps_list_dict, self.camera_poses_np_dict = make_oxford_dataset(root, mode, opt)

    def augment_pc(self, pc_np, intensity_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :param sn_np: 1xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        intensity_np = augmentation.jitter_point_cloud(intensity_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np

    def augment_img(self, img_np):
        """

        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        # color_aug = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
        color_aug =transforms.ColorJitter(brightness=brightness, contrast= contrast, saturation= saturation, hue= hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = augmentation.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random.astype(np.float32)

    def downsample_np(self, pc_np, intensity_np, k):
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



    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, index):

        K = np.asarray([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]], dtype=np.float32)
        K_vis = K
        traversal, pc_timestamp, camera_timestamp, pc_timestamp_idx, traversal_pc_num, traversal_img_num = self.dataset[index]

        pc_timestamps_list = self.pc_timestamps_list_dict[traversal]
        pc_poses_np = self.pc_poses_np_dict[traversal]
        camera_timestamps_list = self.camera_timestamps_list_dict[traversal]
        camera_poses_np = self.camera_poses_np_dict[traversal]
        camera_timestamp_idx = pc_timestamp_idx
        P_o_pc = pc_poses_np[pc_timestamp_idx]
        P_o_cam = camera_poses_np[camera_timestamp_idx]
        P_cam_pc = np.dot(np.linalg.inv(P_o_cam), P_o_pc)

        #load and resize img
        camera_folder = os.path.join(self.root, traversal, 'stereo', 'centre')
        depth_folder = os.path.join(self.root, traversal, 'depth_map')
        camera_timestamp = camera_timestamps_list[camera_timestamp_idx]
        # print(os.path.join(camera_folder, "%d.jpg" % camera_timestamp))
        img = np.array(Image.open(os.path.join(camera_folder, "%d.jpg" % camera_timestamp)))
        img_vis = img
        depth_img = np.array(Image.open(os.path.join(depth_folder, '%d_depth.jpg' % camera_timestamp)))
        # print(traversal, pc_timestamp, pc_timestamp_idx, camera_folder,camera_timestamp ,'load img success')

        # ------------- load image, original size is 960x1280, bottom rows are car itself -------------
        tmp_img_H = img.shape[0]
        img = img[0:(tmp_img_H-self.opt.crop_original_bottom_rows), :, :]
        # scale
        img = cv2.resize(img,
                         (int(round(img.shape[1] * self.opt.img_scale)), int(round((img.shape[0] * self.opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
        depth_img = cv2.resize(depth_img,
                         (int(round(depth_img.shape[1] * self.opt.img_scale)), int(round((depth_img.shape[0] * self.opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)

        # img = Image.fromarray(img)
        K = camera_matrix_scaling(K, self.opt.img_scale)

        # random crop into input size
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        depth_img = np.expand_dims(depth_img, axis=2)
        depth_img = depth_img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        K_4 = camera_matrix_scaling(K, 0.25)

        # ------------- load point cloud ----------------
        if self.opt.is_remove_ground:
            lidar_name = 'lms_front_foreground'
        else:
            lidar_name = 'lms_front'
        pc_path = os.path.join(self.root, traversal, lidar_name, '%d.npy' % pc_timestamp)
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
        pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
        pc_np = pc_np[:, pc_mask_range]
        intensity_np = intensity_np[:, pc_mask_range]
        
        # remove the ground points!

        if pc_np.shape[1] > 2 * self.opt.input_pt_num:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np = downsample_with_reflectance(pc_np, intensity_np[0], voxel_grid_downsample_size=0.2)
            intensity_np = np.expand_dims(intensity_np, axis=0)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
        # random sampling
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np, self.opt.input_pt_num)
        
        #get pc_mask
        pc_cam = np.dot(P_cam_pc[0:3, 0:3], pc_np) + P_cam_pc[0:3, 3:] 
        pc_ = np.dot(K_4, pc_cam)
        depth = pc_[2, :]
        # if(sum(depth==0) > 0):
        #     print(traversal, pc_timestamp)
        pc_mask = np.zeros((1, np.shape(pc_np)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.floor(pc_[0:2, :])
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.opt.img_W * 0.25 - 1)) & (xy[1, :] >= 0) & (
                xy[1, :] <= (self.opt.img_H * 0.25 - 1)) & (pc_[2, :] > 0)
        pc_mask[:, is_in_picture] = 1.

        pc_kpt_idx = np.where(pc_mask.squeeze() == 1)[0]
        index = np.random.permutation(len(pc_kpt_idx))[0:self.opt.num_kpt]
        pc_kpt_idx = pc_kpt_idx[index]

        pc_outline_idx = np.where(pc_mask.squeeze() == 0)[0]
        index = np.random.permutation(len(pc_outline_idx))[0:self.opt.num_kpt]
        pc_outline_idx = pc_outline_idx[index]

       #get img_mask
        xy2 = xy[:, is_in_picture]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])),
                              shape=(int(self.opt.img_H * 0.25), int(self.opt.img_W * 0.25))).toarray()

        img_mask = np.array(img_mask)
        img_mask[img_mask > 0] = 1.

        img_kpt_index = xy[1, pc_kpt_idx] * self.opt.img_W * 0.25 + xy[0, pc_kpt_idx]

        img_outline_index = np.where(img_mask.squeeze().reshape(-1) == 0)[0]
        index = np.random.permutation(len(img_outline_index))[0:self.opt.num_kpt]
        img_outline_index = img_outline_index[index]
        


        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude, self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude, self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            if random.random() > 0.5:
                img = self.augment_img(img)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        else:
            Pr = np.identity(4, dtype=np.float32)
            Pr_inv = np.identity(4, dtype=np.float32)

        P = np.dot(P_cam_pc, Pr_inv)

        # now the point cloud is in CAMERA coordinate
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(Pr, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN


        # ------------ Farthest Point Sampling ------------------
        # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              int(self.opt.node_a_num*8),
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                            int(self.opt.node_b_num*8),
                                                                            replace=False)],
                                                  k=self.opt.node_b_num)



        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np)  # 3xN
        intensity = torch.from_numpy(intensity_np)  # 1xN
        sn = torch.zeros(pc.size(), dtype=pc.dtype, device=pc.device)
        # node_a = torch.from_numpy(node_a_np)  # 3xMa
        # node_b = torch.from_numpy(node_b_np)  # 3xMb

        # P = torch.from_numpy(P[0:3, :].astype(np.float32))  # 3x4
        P = torch.from_numpy(P.astype(np.float32))  # 4x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        depth_img = torch.from_numpy(depth_img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        K = torch.from_numpy(K_4.astype(np.float32))  # 3x3
        # if ((pc_outline_idx.shape[0] < self.opt.num_kpt) | (img_kpt_index.shape[0] < self.opt.num_kpt) | (img_outline_index.shape[0] < self.opt.num_kpt)):
        #     print(traversal, pc_timestamp, pc_timestamp_idx)
        #     print('pc_kpt_idx.shape[0]:',pc_kpt_idx.shape[0]
        #         ,'pc_outline_idx.shape[0]: ',pc_outline_idx.shape[0]
        #         ,'img_kpt_index.shape[0]: ',img_kpt_index.shape[0]
        #         ,'img_outline_index.shape[0]: ',img_outline_index.shape[0])
        #     with open('data_clean_train4.txt', 'a') as file:
        #         file.write(os.path.join(self.root, traversal, lidar_name, '%d.npy' % pc_timestamp) + ',' + os.path.join(camera_folder, "%d.jpg" % camera_timestamp) + '\n')
        # print(traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num, 'done')
        return {'pc': pc,
                'intensity': intensity,
                'sn': sn,
                'P': P,
                'img': img,
                'K': K,

                'pc_mask': torch.from_numpy(pc_mask).float(),
                'img_mask': torch.from_numpy(img_mask).float(),  # (40,128)

                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),  # 512
                'pc_outline_idx': torch.from_numpy(pc_outline_idx),  # 512
                'img_kpt_idx': torch.from_numpy(img_kpt_index).long(),  # 512
                'img_outline_index': torch.from_numpy(img_outline_index).long(),
                'node_a': torch.from_numpy(node_a_np).float(),
                'node_b': torch.from_numpy(node_b_np).float(),
                'depth_img' : depth_img,
                'img_vis' :  torch.from_numpy(img_vis.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'K_vis': torch.from_numpy(K_vis.astype(np.float32)),
                'img_crop_dx' : img_crop_dx,
                'img_crop_dy' : img_crop_dy,
                'seq' : traversal,
                'seq_i': camera_timestamp
                }



if __name__ == '__main__':
    root_path = '/media/ai-i-sunyunda/data/data/oxford_for_i2p'
    opt = options.Options()
    oxfordloader = OxfordLoader(root_path, 'train', opt)
    testdataset = OxfordLoader(root_path, 'val', opt)
    trainloader=torch.utils.data.DataLoader(oxfordloader,batch_size=24,shuffle=True,drop_last=True,num_workers=10)
    testloader = torch.utils.data.DataLoader(testdataset,batch_size=24,shuffle=True,drop_last=True,num_workers=10)
    # print(oxfordloader.dataset[0])
    for i in range(10):
        print(i,'start')
        for step,data in enumerate(trainloader):
            print(step)
        # for i in range(0, len(oxfordloader)):
            # if (step % 100 == 0):
            #     print('----------------------------------------------------------------------%d --------------------------------------------------------------' % step)
        # with open('data_clean_val2.txt', 'a') as file:
        #     file.write('\n')
        # print(data.keys())
        # for item in data.keys():
        #     # print(item.size())
        #     print(item)


