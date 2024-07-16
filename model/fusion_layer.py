import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List
# import sys
# sys.path.append(".")
from kitti.options import Options
from .imagenet import ImageUpSample
from . import layers_pc


class P2I_Fusion(nn.Module):
    def __init__(self, opt:Options):
        super(P2I_Fusion, self).__init__()
        self.opt = opt
        self.img_32_attention_conv = nn.Sequential( nn.Conv2d(512+512, 512, 1, bias=False),
                                                    nn.BatchNorm2d(512), nn.ReLU(),
                                                    nn.Conv2d(512, 512, 1, bias=False),
                                                    nn.BatchNorm2d(512), nn.ReLU(),
                                                    nn.Conv2d(512, self.opt.node_b_num, 1, bias=False))

        self.img_16_attention_conv = nn.Sequential( nn.Conv2d(512+256, 256, 1, bias=False),
                                                    nn.BatchNorm2d(256), nn.ReLU(),
                                                    nn.Conv2d(256, 256, 1, bias=False),
                                                    nn.BatchNorm2d(256), nn.ReLU(),
                                                    nn.Conv2d(256, self.opt.node_a_num, 1, bias=False))
        self.up_conv1 = ImageUpSample(768+320,256)
        self.up_conv2 = ImageUpSample(256+128,128)
        self.up_conv3 = ImageUpSample(128+64+64,64)


    def forward(self, img_s32_feature_map, img_s16_feature_map, img_s8_feature_map, img_s4_feature_map, img_s2_feature_map, global_feature, node_b_features, node_a_features):
        
        B, C, H, W = img_s32_feature_map.size()


        img_s32_feature_map_pc_global_feature = torch.cat((img_s32_feature_map, global_feature.unsqueeze(-1).expand(B, global_feature.size(1), H, W)), dim=1)
        img_32_attention = self.img_32_attention_conv(img_s32_feature_map_pc_global_feature)
        img_32_attention = F.softmax(img_32_attention, dim=1)
        img_s32_feature_map_fusion = torch.cat((torch.sum(img_32_attention.unsqueeze(1) * node_b_features.unsqueeze(-1).unsqueeze(-1), dim=2), img_s32_feature_map), dim=1)


        img_s16_feature_map_pc_global_feature = torch.cat((img_s16_feature_map, global_feature.unsqueeze(-1).expand(B, global_feature.size(1), img_s16_feature_map.size(-2), img_s16_feature_map.size(-1))), dim=1)
        img_16_attention = self.img_16_attention_conv(img_s16_feature_map_pc_global_feature)
        img_16_attention = F.softmax(img_16_attention, dim=1)
        img_s16_feature_map_fusion = torch.cat((torch.sum(img_16_attention.unsqueeze(1) * node_a_features.unsqueeze(-1).unsqueeze(-1), dim=2), img_s16_feature_map), dim=1)

        image_feature_16=self.up_conv1(img_s32_feature_map_fusion,img_s16_feature_map_fusion)
        image_feature_8=self.up_conv2(image_feature_16,img_s8_feature_map)
        img_s4_feature_map=torch.cat((img_s4_feature_map,F.interpolate(img_s2_feature_map,scale_factor=0.5)),dim=1)
        image_feature_mid=self.up_conv3(image_feature_8,img_s4_feature_map)

        return image_feature_mid

class I2P_Fusion(nn.Module):
    def __init__(self, opt:Options):
        super(I2P_Fusion, self).__init__()
        self.opt = opt
        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))
        self.node_b_attention_pn = layers_pc.PointNet(256+512,
                                                    [256, self.H_fine_res*self.W_fine_res],
                                                    activation=self.opt.activation,
                                                    normalization=self.opt.normalization,
                                                    norm_momentum=self.opt.norm_momentum,
                                                    norm_act_at_last=False)
        self.node_b_pn = layers_pc.PointNet(256+512+512+512,
                                            [1024, 512, 512],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=self.opt.norm_momentum,
                                            norm_act_at_last=False)
        self.node_a_attention_pn = layers_pc.PointNet(64 + 512,
                                                      [256, int(self.H_fine_res * self.W_fine_res * 4)],
                                                      activation=self.opt.activation,
                                                      normalization=self.opt.normalization,
                                                      norm_momentum=self.opt.norm_momentum,
                                                      norm_act_at_last=False)
        self.node_a_pn = layers_pc.PointNet(64+256+512,
                                            [512, 128, 128],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=self.opt.norm_momentum,
                                            norm_act_at_last=False)

    def gather_topk_features(self, min_k_idx, features):
        """

        :param min_k_idx: BxNxk
        :param features: BxCxM
        :return:
        """
        B, N, k = min_k_idx.size(0), min_k_idx.size(1), min_k_idx.size(2)
        C, M = features.size(1), features.size(2)

        return torch.gather(features.unsqueeze(3).expand(B, C, M, k),
                            index=min_k_idx.unsqueeze(1).expand(B, C, N, k),
                            dim=2)  # BxCxNxk

    def upsample_by_interpolation(self,
                                  interp_ab_topk_idx,
                                  node_a,
                                  node_b,
                                  up_node_b_features):
        interp_ab_topk_node_b = self.gather_topk_features(interp_ab_topk_idx, node_b)  # Bx3xMaxk
        # Bx3xMa -> Bx3xMaxk -> BxMaxk
        interp_ab_node_diff = torch.norm(node_a.unsqueeze(3) - interp_ab_topk_node_b, dim=1, p=2, keepdim=False)
        interp_ab_weight = 1 - interp_ab_node_diff / torch.sum(interp_ab_node_diff, dim=2, keepdim=True)  # BxMaxk
        interp_ab_topk_node_b_features = self.gather_topk_features(interp_ab_topk_idx, up_node_b_features)  # BxCxMaxk
        # BxCxMaxk -> BxCxMa
        interp_ab_weighted_node_b_features = torch.sum(interp_ab_weight.unsqueeze(1) * interp_ab_topk_node_b_features,
                                                       dim=3)
        return interp_ab_weighted_node_b_features

    def forward(self, pc, node_a, node_b,
                img_global_feature_BCMa, img_global_feature_BCMb, img_s32_feature_map_BCHw, img_s16_feature_map_BCHw,
                node_b_features, global_feature, node_a_features, node_a_min_k_idx):
        
        B, Mb = pc.size(0), node_b.size(2)
        C_global = global_feature.size(1)

        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb

        node_b_attention_score=F.softmax(node_b_attention_score,dim=1)

        node_b_weighted_img_s32_feature_map = torch.sum(img_s32_feature_map_BCHw.unsqueeze(3) * node_b_attention_score.unsqueeze(1),
                                                  dim=2)  # BxC_imgx(H*W)xMb -> BxC_imgxMb
        up_node_b_features = self.node_b_pn(torch.cat((node_b_features,
                                                       global_feature.expand(B, C_global, Mb),
                                                       node_b_weighted_img_s32_feature_map,
                                                       img_global_feature_BCMb), dim=1))  # BxCxMb

        # interpolation of pc over node_b
        pc_node_b_diff = torch.norm(pc.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxNxMb
        # BxNxk
        _, interp_pc_node_b_topk_idx = torch.topk(pc_node_b_diff, k=self.opt.k_interp_point_b,
                                                  dim=2, largest=False, sorted=True)
        interp_pb_weighted_node_b_features = self.upsample_by_interpolation(interp_pc_node_b_topk_idx,
                                                                            pc,
                                                                            node_b,
                                                                            up_node_b_features)


        # interpolation of point over node_a  ----------------------------------------------
        # use attention method to select resnet features for each node_a_feature
        node_a_attention_score = self.node_a_attention_pn(torch.cat((node_a_features,
                                                                     img_global_feature_BCMa), dim=1))  # Bx(H*W)xMa
        
        node_a_attention_score=F.softmax(node_a_attention_score,dim=1)
        
        node_a_weighted_img_s16_feature_map = torch.sum(
            img_s16_feature_map_BCHw.unsqueeze(3) * node_a_attention_score.unsqueeze(1),
            dim=2)  # BxC_imgx(H*W)xMa -> BxC_imgxMa
        # interpolation of node_a over node_b
        node_a_node_b_diff = torch.norm(node_a.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxMaxMb
        _, interp_nodea_nodeb_topk_idx = torch.topk(node_a_node_b_diff, k=self.opt.k_interp_ab,
                                                    dim=2, largest=False, sorted=True)
        interp_ab_weighted_node_b_features = self.upsample_by_interpolation(interp_nodea_nodeb_topk_idx,
                                                                            node_a,
                                                                            node_b,
                                                                            up_node_b_features)

        up_node_a_features = self.node_a_pn(torch.cat((node_a_features,
                                                       interp_ab_weighted_node_b_features,
                                                       node_a_weighted_img_s16_feature_map),
                                                      dim=1))  # BxCxMa
        interp_pa_weighted_node_a_features = self.upsample_by_interpolation(node_a_min_k_idx,
                                                                            pc,
                                                                            node_a,
                                                                            up_node_a_features)

        return interp_pa_weighted_node_a_features, interp_pb_weighted_node_b_features, 


class Img_Point_Fusion_Net(nn.Module):
    def __init__(self, opt:Options):
        self.opt = opt
        super(Img_Point_Fusion_Net,self).__init__()
        self.i2p_fusion = P2I_Fusion(opt)
        self.p2i_fusion = I2P_Fusion(opt)

    def forward(self, pc, node_a, node_b,
                img_s32_feature_map, img_s16_feature_map, 
                img_s8_feature_map, img_s4_feature_map,
                img_s2_feature_map, img_global_feature,
                global_feature, node_b_features, node_a_features, node_a_min_k_idx):

        B,Ma,Mb = pc.size(0), node_a.size(2), node_b.size(2)
        C_img = img_global_feature.size(1)

        P2I_feature_set = self.i2p_fusion(img_s32_feature_map, 
                                        img_s16_feature_map, 
                                        img_s8_feature_map, 
                                        img_s4_feature_map, 
                                        img_s2_feature_map, 
                                        global_feature, 
                                        node_b_features, 
                                        node_a_features)

        img_s16_feature_map_BCHw = img_s16_feature_map.view(B,img_s16_feature_map.size(1),-1)
        img_s32_feature_map_BCHw = img_s32_feature_map.view(B,img_s32_feature_map.size(1),-1)
        img_global_feature_BCMa = img_global_feature.squeeze(3).expand(B, C_img, Ma)  # BxC_img -> BxC_imgxMa
        img_global_feature_BCMb = img_global_feature.squeeze(3).expand(B, C_img, Mb)  # BxC_img -> BxC_imgxMb



        I2P_feature_set = self.p2i_fusion(pc, 
                                        node_a, node_b,
                                        img_global_feature_BCMa,
                                        img_global_feature_BCMb, 
                                        img_s32_feature_map_BCHw, 
                                        img_s16_feature_map_BCHw,
                                        node_b_features,  
                                        global_feature, 
                                        node_a_features, 
                                        node_a_min_k_idx)


        return P2I_feature_set, I2P_feature_set

