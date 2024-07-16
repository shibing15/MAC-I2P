import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers_pc
from . import imagenet
from . import pointnet2
from kitti.options import Options
from .fusion_layer import Img_Point_Fusion_Net
from .spvcnn.spvcnn import SPVCNN

class MAI2P(nn.Module):
    def __init__(self, opt:Options):
        super(MAI2P, self).__init__()
        
        self.opt = opt
        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        #3D backbone
        self.pc_encoder = pointnet2.PCEncoder(opt, Ca=64, Cb=256, Cg=512)
        self.voxel_branch = SPVCNN(num_classes=128, cr=0.5, pres=0.05, vres=0.05)
       
        #2D backbone
        self.img_encoder = imagenet.ImageEncoder()
       
        #Feature Fusion
        self.pc_img_fusion_net = Img_Point_Fusion_Net(opt)
        
        per_point_pn_in_channels = 32 + 64 + 128 + 512
        self.per_point_pn=layers_pc.PointNet(per_point_pn_in_channels,
                                            [256, 256, 128],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=True,)
        per_point_pn_in_channels_cla = 32 + 64 + 128 + 512 + 128
        self.point_classifer = layers_pc.PointNet(per_point_pn_in_channels_cla,
                                                [256, 256, self.H_fine_res * self.W_fine_res],
                                                activation=self.opt.activation,
                                                normalization=self.opt.normalization,
                                                norm_momentum=opt.norm_momentum,
                                                norm_act_at_last=False,
                                                dropout_list=[0.5, 0.5, 0])
        self.pc_feature_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False),nn.BatchNorm1d(64))
        self.pc_score_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,1,1,bias=False),nn.Sigmoid())                                                

        self.img_feature_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False), nn.BatchNorm2d(64))
        self.img_score_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,bias=False),nn.Sigmoid())
        


    def forward(self, pc, intensity, sn, img, node_a, node_b):
        
        #para inital
        B,N,Ma,Mb=pc.size(0),pc.size(2),node_a.size(2),node_b.size(2)

        #feature extract
        pc_center,\
        cluster_mean, \
        node_a_min_k_idx, \
        first_pn_out, \
        second_pn_out, \
        node_a_features, \
        node_b_features, \
        global_feature = self.pc_encoder(pc,
                                          intensity,
                                          sn,
                                          node_a,
                                          node_b)
        #-----------------------voxel_pc------------------------------------------------------
        input_voxel = torch.cat((pc, intensity, sn), dim=1).transpose(2,1).reshape(-1, 7)
        batch_inds = torch.arange(pc.shape[0]).reshape(-1,1).repeat(1,pc.shape[2]).reshape(-1, 1).cuda()
        corrds = pc.transpose(2,1) - torch.min(pc.transpose(2,1), dim=1, keepdim=True)[0]
        corrds = corrds.reshape(-1, 3)
        corrds = torch.round(corrds / 0.05)
        corrds = torch.cat((corrds, batch_inds), dim=-1)
        _, voxel_feat = self.voxel_branch(input_voxel, corrds, pc.shape[0])

        img_feature_set=self.img_encoder(img)

        C_global = global_feature.size(1)

        img_global_feature=img_feature_set[-1]  #512
        img_s32_feature_map=img_feature_set[-2] #512
        img_s16_feature_map=img_feature_set[-3] #256
        img_s8_feature_map=img_feature_set[-4]  #128
        img_s4_feature_map=img_feature_set[-5]  #64
        img_s2_feature_map=img_feature_set[-6]  #64

        #---------------------------------fusion-------------------------
        P2I_feature_set, I2P_feature_set = self.pc_img_fusion_net(pc, node_a, node_b,
                                                img_s32_feature_map, img_s16_feature_map,
                                                img_s8_feature_map, img_s4_feature_map,
                                                img_s2_feature_map, img_global_feature,
                                                global_feature, node_b_features, node_a_features, node_a_min_k_idx)

        img_feature = self.img_feature_layer(P2I_feature_set)

        img_score = self.img_score_layer(P2I_feature_set)


        pc_label_scores = self.per_point_pn(torch.cat((I2P_feature_set[0], 
                                                I2P_feature_set[1], 
                                                first_pn_out, 
                                                second_pn_out), dim=1))
        pc_class_scores = self.point_classifer(torch.cat((I2P_feature_set[0], 
                                                I2P_feature_set[1], 
                                                first_pn_out, 
                                                second_pn_out,
                                                voxel_feat), dim=1))

        pc_feature = self.pc_feature_layer(pc_label_scores)

        pc_score = self.pc_score_layer(pc_label_scores)

        img_feature_flatten = img_feature.flatten(start_dim=2)

        pc_feature_norm=F.normalize(pc_feature, dim=1,p=2)
        img_feature_norm=F.normalize(img_feature_flatten, dim=1,p=2)



        return img_feature_norm, pc_feature_norm, \
               img_score,pc_score, pc_class_scores
if __name__=='__main__':
    opt=Options()
    pc=torch.rand(10,3,20480).cuda()
    intensity=torch.rand(10,1,20480).cuda()
    sn=torch.rand(10,3,20480).cuda()
    img=torch.rand(10,3,160,512).cuda()
    net=MAI2P(opt).cuda()
    a,b,c,d=net(pc,intensity,sn,img)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())

    