import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import argparse
from model.network import MAI2P

from kitti.kitti_pc_img_dataloader import kitti_pc_img_dataset
from kitti import options
# import loss
from model.loss import  cw_loss, match_loss
import numpy as np
import logging
import math
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff


def test_acc(model,testdataloader,args):
    
    t_diff_set=[]
    angles_diff_set=[]
    t_val_diff_set = []
    angles_val_diff_set = []
    success_rate = 0
    for step,data in enumerate(testdataloader):
        if step%30==0:
            model.eval()
            img_=data['img'].cuda()              #full size
            depth_img = data['depth_img'].cuda()
            pc=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()      
            img_mask=data['img_mask'].cuda()    #1/4 size

            pc_kpt_idx=data['pc_kpt_idx'].cuda()                #(B,512)
            pc_outline_idx=data['pc_outline_idx'].cuda()
            img_kpt_idx=data['img_kpt_idx'].cuda()
            img_outline_idx=data['img_outline_index'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()

            img = torch.cat((img_, depth_img), dim=1)
            img_siam_feature_norm, pc_siam_feature_norm, \
            img_score,pc_score=model(pc,intensity,sn,img,node_a,node_b)     #64 channels feature
            
            img_score=img_score[0].data.cpu().numpy()
            pc_score=pc_score[0].data.cpu().numpy()
            img_siam_feature=img_siam_feature_norm[0].data.cpu().numpy()
            pc_feature=pc_siam_feature_norm[0].data.cpu().numpy()
            pc=pc[0].data.cpu().numpy()
            P=P[0].data.cpu().numpy()
            K=K[0].data.cpu().numpy()
            
            img_x=np.linspace(0,np.shape(img_score)[-1]-1,np.shape(img_score)[-1]).reshape(1,-1).repeat(np.shape(img_score)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
            img_y=np.linspace(0,np.shape(img_score)[-2]-1,np.shape(img_score)[-2]).reshape(-1,1).repeat(np.shape(img_score)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])

            img_xy=np.concatenate((img_x,img_y),axis=0)

            img_xy_flatten=img_xy.reshape(2,-1)
            # img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
            img_feature_flatten = img_siam_feature
            img_score_flatten=img_score.squeeze().reshape(-1)

            img_index=(img_score_flatten>args.img_thres)
            #topk_img_index=np.argsort(-img_score_flatten)[:args.num_kpt]
            img_xy_flatten_sel=img_xy_flatten[:,img_index]
            img_feature_flatten_sel=img_feature_flatten[:,img_index]
            img_score_flatten_sel=img_score_flatten[img_index]

            pc_index=(pc_score.squeeze()>args.pc_thres)
            #topk_pc_index=np.argsort(-pc_score.squeeze())[:args.num_kpt]
            pc_sel=pc[:,pc_index]
            pc_feature_sel=pc_feature[:,pc_index]
            pc_score_sel=pc_score.squeeze()[pc_index]


            dist= np.sum(np.expand_dims(pc_feature_sel,axis=2)*np.expand_dims(img_feature_flatten_sel,axis=1),axis=0)
            sel_index=np.argsort(-dist,axis=0)[0,:]

            # img_xy_pc=img_xy_flatten_sel[:,sel_index]
            img_xy_pc = img_xy_flatten_sel
            pc_sel = pc_sel[:, sel_index]

            is_success,R,t,inliers=cv2.solvePnPRansac(pc_sel.T,img_xy_pc.T,K,useExtrinsicGuess=False,
                                                        iterationsCount=500,
                                                        reprojectionError=args.dist_thres,
                                                        flags=cv2.SOLVEPNP_EPNP,
                                                        distCoeffs=None)
            R,_=cv2.Rodrigues(R)
            T_pred=np.eye(4)
            T_pred[0:3,0:3]=R
            T_pred[0:3,3:]=t
            t_diff,angles_diff=get_P_diff(T_pred,P)
            if (t_diff < 5) & (angles_diff < 10):
                t_val_diff_set.append(t_diff)
                angles_val_diff_set.append(angles_diff)
            t_diff_set.append(t_diff)
            angles_diff_set.append(angles_diff)
    success_rate = len(t_val_diff_set) / len(t_diff_set)
    return np.mean(np.array(t_diff_set)),np.mean(np.array(angles_diff_set)), np.mean(np.array(t_val_diff_set)), np.mean(np.array(angles_val_diff_set)), success_rate


def model_inference(model,data,args):
    img_=data['img'].cuda()
    depth_img = data['depth_img'].cuda()                  #full size
    pc=data['pc'].cuda()
    intensity=data['intensity'].cuda()
    sn=data['sn'].cuda()
    K=data['K'].cuda()
    P=data['P'].cuda()
    pc_mask=data['pc_mask'].cuda()      
    img_mask=data['img_mask'].cuda()        #1/4 size
    pc_kpt_idx=data['pc_kpt_idx'].cuda()    #(B,512)
    pc_outline_idx=data['pc_outline_idx'].cuda()
    img_kpt_idx=data['img_kpt_idx'].cuda()
    img_outline_idx=data['img_outline_index'].cuda()
    node_a=data['node_a'].cuda()
    node_b=data['node_b'].cuda()
    img_x=torch.linspace(0,img_mask.size(-1)-1,img_mask.size(-1)).view(1,-1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
    img_y=torch.linspace(0,img_mask.size(-2)-1,img_mask.size(-2)).view(-1,1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
    img_xy=torch.cat((img_x,img_y),dim=1)


    #point class inital
    B, img_H, img_W  = img_.size(0), img_.size(2), img_.size(3)
    N = pc.size(2)
    img_W_fine_res = int(round(img_W / opt.img_fine_resolution_scale))
    img_H_fine_res = int(round(img_H / opt.img_fine_resolution_scale))

      
    img = torch.cat((img_, depth_img), dim=1)
    img_siam_feature_norm, pc_siam_feature_norm, \
    img_score,pc_score, pc_class_scores = model(pc,intensity,sn,img,node_a,node_b)    #64 channels feature


    
    pc_features_inline=torch.gather(pc_siam_feature_norm,index=pc_kpt_idx.unsqueeze(1).expand(B,pc_siam_feature_norm.size(1),args.num_kpt),dim=-1)    #(B,C,num_kpt)
    pc_features_outline=torch.gather(pc_siam_feature_norm,index=pc_outline_idx.unsqueeze(1).expand(B,pc_siam_feature_norm.size(1),args.num_kpt),dim=-1)
    pc_xyz_inline=torch.gather(pc,index=pc_kpt_idx.unsqueeze(1).expand(B,3,args.num_kpt),dim=-1)
    pc_score_inline=torch.gather(pc_score,index=pc_kpt_idx.unsqueeze(1),dim=-1)         #(B,1,num_in)
    pc_score_outline=torch.gather(pc_score,index=pc_outline_idx.unsqueeze(1),dim=-1)    #(B,1,num_out)

            
    # img_features_flatten=img_features.contiguous().view(img_features.size(0),img_features.size(1),-1)   #(B,C,(H*W))

    img_features_flatten = img_siam_feature_norm
    img_score_flatten=img_score.contiguous().view(img_score.size(0),img_score.size(1),-1)               #(B,1,(H*W))
    img_xy_flatten=img_xy.contiguous().view(img_siam_feature_norm.size(0),2,-1)
    img_features_flatten_inline=torch.gather(img_features_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),args.num_kpt),dim=-1)
    img_xy_flatten_inline=torch.gather(img_xy_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,2,args.num_kpt),dim=-1)
    img_score_flatten_inline=torch.gather(img_score_flatten,index=img_kpt_idx.unsqueeze(1),dim=-1)
    img_features_flatten_outline=torch.gather(img_features_flatten,index=img_outline_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),args.num_kpt),dim=-1)
    img_score_flatten_outline=torch.gather(img_score_flatten,index=img_outline_idx.unsqueeze(1),dim=-1)
    

    #----------------------------------------------cal_point_class_loss--------------------------------
    img_W_for_pred = int(img_W * 0.25)
    img_H_for_pred = int(img_H * 0.25)
    # print(img_W_for_pred)
    # print(img_H_for_pred)
    pc_px_py_pz=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc)+P[:,0:3,3:]))
    pc_uv = pc_px_py_pz[:,0:2,:]/pc_px_py_pz[:,2:,:]
    x_inside_mask = (pc_uv[:, 0:1, :] >= 0) \
                    & (pc_uv[:, 0:1, :] <= img_W_for_pred - 1)  # Bx1xN
    y_inside_mask = (pc_uv[:, 1:2, :] >= 0) \
                    & (pc_uv[:, 1:2, :] <= img_H_for_pred - 1)  # Bx1xN
    z_inside_mask = pc_px_py_pz[:, 2:3, :] > 0.1  # Bx1xN
    inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN

    KP_pc_pxpy_scale_int = torch.floor(pc_uv / 8.0).to(dtype=torch.long)
    KP_pc_pxpy_index = KP_pc_pxpy_scale_int[:, 0, :] + KP_pc_pxpy_scale_int[:, 1, :] * int(round(img_W / opt.img_fine_resolution_scale))  # BxN

    # get fine labels
    # organize everything into (B*N)x* shape
    inside_mask_Bn = inside_mask.reshape(B*N)  # BN
    inside_mask_Bn_int = inside_mask_Bn.to(dtype=torch.int32)  # BN
    insider_num = int(torch.sum(inside_mask_Bn_int).item())  # scalar
    _, inside_idx_Bn = torch.sort(inside_mask_Bn_int, descending=True)  # BN
    insider_idx = inside_idx_Bn[0: insider_num]  # B_insider
    
    KP_pc_pxpy_index_Bn = KP_pc_pxpy_index.view(B*N)  # BN in long
    KP_pc_pxpy_index_insider = torch.gather(KP_pc_pxpy_index_Bn, dim=0, index=insider_idx)  # B_insider in long
    # assure correctness
    fine_labels_min = torch.min(KP_pc_pxpy_index_insider).item()
    fine_labels_max = torch.max(KP_pc_pxpy_index_insider).item()
    assert fine_labels_min >= 0
    assert fine_labels_max <= img_W_fine_res * img_H_fine_res - 1
    # BxLxN -> BxNxL
    L = pc_class_scores.size(1)
    fine_scores_BnL = pc_class_scores.permute(0, 2, 1).reshape(B*N, L).contiguous()  # BNxL
    insider_idx_BinsiderL = insider_idx.unsqueeze(1).expand(insider_num, L)  # B_insiderxL
    fine_scores_insider = torch.gather(fine_scores_BnL, dim=0, index=insider_idx_BinsiderL)  # B_insiderxL
    #----------------cal_coview_loss and pixel-point match loss------------------
    pc_xyz_projection=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc_xyz_inline)+P[:,0:3,3:]))
    pc_xy_projection=pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]

    correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=1))<=args.dist_thres).float()

    #-----------------------cal_loss-----------------------------
    loss_pc_cla = loss_pc_class(fine_scores_insider, KP_pc_pxpy_index_insider)
    #根据features计算loss
    siamese_loss = match_loss(img_features_flatten_inline, pc_features_inline, correspondence_mask, 1.0)

    loss_det=cw_loss(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline.squeeze(),pc_score_outline.squeeze())

    loss_dict = {'loss_pc_cla': loss_pc_cla,
                 'siamese_loss': siamese_loss,
                 'loss_det': loss_det}

    #----------------------------------cal_acc----------------------------------------------
    _, fine_predictions_insider = torch.max(fine_scores_insider, dim=1, keepdim=False)
    pc_class_accuracy = torch.sum(torch.eq(KP_pc_pxpy_index_insider, fine_predictions_insider).to(dtype=torch.float)) / insider_num

    pc_scores_flatten = pc_score.permute(0, 2, 1).squeeze(-1).contiguous()  # BNxL
    pc_coview_pre = torch.where(pc_scores_flatten > 0.95, 1.0, 0.0)
    pc_coview_accuracy = torch.sum(torch.eq(inside_mask.to(dtype=torch.long), pc_coview_pre).to(dtype=torch.float)) / ( B * N)

    acc_dict = {'pc_class_accuracy': pc_class_accuracy,
                'pc_coview_accuracy': pc_coview_accuracy}


    return loss_dict, acc_dict


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--epoch', type=int, default=25, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=8, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='', metavar='data_path', help='train and test data path')
    parser.add_argument('--num_point', type=int, default=20480, metavar='num_point',
                        help='point cloud size to train')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='')
    parser.add_argument('--min_lr', type=float, default=0.00001, metavar='lr',
                        help='')
    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=2*math.pi*0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=2*math.pi*0, metavar='P_Rz_amplitude',
                        help='')
    parser.add_argument('--save_path', type=str, default='./runs/log_xy_20480_128_kitti', metavar='save_path',
                        help='path to save log and model')
    '''parser.add_argument('--save_path', type=str, default='./only_test', metavar='save_path',
                        help='path to save log and model')'''
    parser.add_argument('--num_kpt', type=int, default=512, metavar='num_kpt',
                        help='')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='num_kpt',
                        help='')
    parser.add_argument('--img_thres', type=float, default=0.95, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.95, metavar='pc_thres',
                        help='')
    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logdir=os.path.join(args.save_path, 'dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(args.dist_thres,args.pos_margin,args.neg_margin,))
    try:
        os.makedirs(logdir)
    except:
        print('mkdir failue')

    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (logdir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    train_dataset = kitti_pc_img_dataset(args.data_path, 'train', args.num_point,
                                         P_tx_amplitude=args.P_tx_amplitude,
                                         P_ty_amplitude=args.P_ty_amplitude,
                                         P_tz_amplitude=args.P_tz_amplitude,
                                         P_Rx_amplitude=args.P_Rx_amplitude,
                                         P_Ry_amplitude=args.P_Ry_amplitude,
                                         P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.num_point,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
    assert len(train_dataset) > 10
    assert len(test_dataset) > 10
    trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,drop_last=True,num_workers=args.num_workers)
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    opt=options.Options()
    model=MAI2P(opt)
    checkpoints = torch.load('./runs/kitti/kitti_mmi2p.pth')
    model.load_state_dict(checkpoints)
    model=model.cuda()

    current_lr=args.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)
    logger.info(args)

    global_step=0

    best_t_diff=1000
    best_r_diff=1000
    best_test_accuracy = 0

    loss_pc_class = nn.CrossEntropyLoss()
    for epoch in range(args.epoch):

        for step,data in enumerate(trainloader):
            global_step+=1
            model.train()
            optimizer.zero_grad()
            train_loss_dict, train_acc_dict = model_inference(model, data, args)

            #Unpack loss_dict
            loss_pc_cla = train_loss_dict['loss_pc_cla']
            siamese_loss = train_loss_dict['siamese_loss']
            loss_det = train_loss_dict['loss_det']

            # loss = siamese_loss * 4  + loss_det * 2 + loss_pc_cla
            loss = siamese_loss * 2  + loss_det * 2 + loss_pc_cla
            loss.backward()
            optimizer.step()
            
            #Unpack acc_dict
            pc_class_accuracy = train_acc_dict['pc_class_accuracy']
            pc_coview_accuracy = train_acc_dict['pc_coview_accuracy']

            #----------------------------------------------------------------------------------------

            if global_step%args.train_batch_size==0:
                logger.info('%s-%d-%d, loss: %f, siamese_loss: %f, loss det: %f, loss_pc_cla: %f, pc_class_acc: %f, pc_coview_acc: %f'%('train',epoch,global_step,loss.data.cpu().numpy(),siamese_loss.data.cpu().numpy(),loss_det.data.cpu().numpy(), loss_pc_cla.data.cpu().numpy(), pc_class_accuracy.data.cpu().numpy(), pc_coview_accuracy.data.cpu().numpy()))
                # logger.info('%s-%d-%d, loss: %f, siamese_loss: %f, loss det: %f'%('train',epoch,global_step,loss.data.cpu().numpy(),siamese_loss.data.cpu().numpy(),loss_det.data.cpu().numpy()))

        #epoch done
        test_batch_sum = 0
        test_loss_sum = {'loss_pc_cla': 0,
                        'siamese_loss': 0,
                        'loss_det': 0}
        test_accuracy_sum = {'coview_acc': 0, 'pc_class_acc': 0}
        test_num = 0
        test_loop = tqdm(enumerate(testloader), total= len(testloader))
        for i, data in test_loop:
            test_num += 1
            model.eval()
            with torch.no_grad():
                test_loss_dict, test_acc_dict = model_inference(model, data, args)
            test_batch_sum += args.val_batch_size
            test_loss_sum['loss_pc_cla'] += args.val_batch_size * test_loss_dict['loss_pc_cla']
            test_loss_sum['siamese_loss'] += args.val_batch_size * test_loss_dict['siamese_loss']
            test_loss_sum['loss_det'] += args.val_batch_size * test_loss_dict['loss_det']
            test_accuracy_sum['coview_acc'] += args.val_batch_size * test_acc_dict['pc_coview_accuracy']
            test_accuracy_sum['pc_class_acc'] += args.val_batch_size * test_acc_dict['pc_class_accuracy']
        
        test_loss_sum['loss_pc_cla'] /= test_batch_sum
        test_loss_sum['siamese_loss'] /= test_batch_sum
        test_loss_sum['loss_det'] /= test_batch_sum
        test_accuracy_sum['coview_acc'] /= test_batch_sum
        test_accuracy_sum['pc_class_acc'] /= test_batch_sum
        logger.info('%s-%d, test_siamese_loss: %f, test_loss det: %f, test_loss_pc_cla: %f, test_pc_class_acc: %f, test_pc_coview_acc: %f'%('test',epoch,test_loss_sum['siamese_loss'].data.cpu().numpy(),test_loss_sum['loss_det'].data.cpu().numpy(), test_loss_sum['loss_pc_cla'].data.cpu().numpy(), test_accuracy_sum['pc_class_acc'].data.cpu().numpy(), test_accuracy_sum['coview_acc'].data.cpu().numpy()))

        # record best test loss
        if test_accuracy_sum['pc_class_acc'] > best_test_accuracy:
            best_test_accuracy = test_accuracy_sum['pc_class_acc']
            logger.info('--- best test coarse accuracy %f' % best_test_accuracy)

        
        if epoch%5==0 and epoch>5:
            current_lr=current_lr*0.25
            if current_lr<args.min_lr:
                current_lr=args.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr']=current_lr
            logger.info('%s-%d-%d, updata lr, current lr is %f'%('train',epoch,global_step,current_lr))
        torch.save(model.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))