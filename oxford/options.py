import numpy as np
import math
import torch
import random

class Options:
    def __init__(self):
        self.dataroot = '/media/ai-i-sunyunda/data/data/oxford_for_i2p'
        # self.dataroot = '/data/personal/jiaxin/datasets/kitti'
        self.checkpoints_dir = 'checkpoints'
        self.version = '3.3'
        self.is_debug = False
        self.is_fine_resolution = True
        self.is_remove_ground = False
        self.accumulation_frame_num = 3 #3
        self.accumulation_frame_skip = 4

        self.pc_build_interval = 2
        self.crop_original_bottom_rows = 0
        self.pc_min_range = -1.0
        self.pc_max_range = 50.0

        self.translation_max = 10.0
        self.test_translation_max = 10.0
        self.range_radius = 100
        self.crop_original_top_rows = 100
        self.img_scale = 0.5
        self.img_H = 384  # after scale before crop 800 * 0.4 = 320
        self.img_W = 640  # after scale before crop 1600 * 0.4 = 640
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.num_kpt=512

        self.input_pt_num = 40960
        self.node_a_num = 256
        # self.node_a_num = 128
        self.node_b_num = 256
        # self.node_b_num = 128
        self.k_ab = 32
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # ENU coordinate
        self.P_tx_amplitude = self.translation_max
        self.P_ty_amplitude = self.translation_max * 0.5
        self.P_tz_amplitude = self.translation_max
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0.0 * math.pi / 12.0
        self.dataloader_threads = 10

        self.batch_size = 12
        self.gpu_ids = [1]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.001
        self.lr_decay_step = 15
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4
        if self.is_fine_resolution:
            self.coarse_loss_alpha = 50
        else:
            self.coarse_loss_alpha = 1




if __name__=='__main__':
    camera_name_list = ['CAM_FRONT',
                        # 'CAM_FRONT_LEFT',
                        # 'CAM_FRONT_RIGHT',
                        # 'CAM_BACK',
                        # 'CAM_BACK_LEFT',
                        # 'CAM_BACK_RIGHT'
                        ]
    for i in range(100):
        print(i,random.choice(camera_name_list))