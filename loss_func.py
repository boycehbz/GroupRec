'''
 @FileName    : loss_func.py
 @EditTime    : 2022-01-13 19:16:39
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : define loss functions here
'''

import torch.nn as nn
import torch
import numpy as np
from utils.geometry import batch_rodrigues

class L1(nn.Module):
    def __init__(self, device):
        super(L1, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss(size_average=False)

    def forward(self, x, y):
        b = x.shape[0]
        diff = self.L1Loss(x, y)
        diff = diff / b
        return diff

class SMPL_Loss(nn.Module):
    def __init__(self, device):
        super(SMPL_Loss, self).__init__()
        self.device = device
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.beta_loss_weight = 0.001
        self.pose_loss_weight = 1.0

    def forward(self, pred_rotmat, gt_pose, pred_betas, gt_betas, has_smpl):
        loss_dict = {}
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]

        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]

        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['pose_Loss'] = loss_regr_pose * self.pose_loss_weight
        loss_dict['shape_Loss'] = loss_regr_betas * self.beta_loss_weight
        return loss_dict


class Keyp_Loss(nn.Module):
    def __init__(self, device):
        super(Keyp_Loss, self).__init__()
        self.device = device
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.keyp_weight = 10.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_keypoints_2d, gt_keypoints_2d):
        loss_dict = {}
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        pred_keypoints_2d = pred_keypoints_2d[:,self.halpe2lsp]
        gt_keypoints_2d = gt_keypoints_2d[:,self.halpe2lsp]

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()

        if loss > 300:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['keyp_Loss'] = loss * self.keyp_weight
        return loss_dict

class Mesh_Loss(nn.Module):
    def __init__(self, device):
        super(Mesh_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss().to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0

    def forward(self, pred_vertices, gt_vertices, has_smpl):
        loss_dict = {}
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]

        if len(gt_vertices_with_shape) > 0:
            vert_loss = self.criterion_vert(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            vert_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['vert_loss'] = vert_loss * self.verts_weight
        return loss_dict

class Skeleton_Loss(nn.Module):
    def __init__(self, device):
        super(Skeleton_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.skeleton_weight = 5.0
        self.verts_weight = 5.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.right_start = [12, 8, 7, 12, 2, 1]
        self.right_end = [8, 7, 6, 2, 1, 0]
        self.left_start = [12, 9, 10, 12, 3, 4]
        self.left_end = [9, 10, 11, 3, 4, 5]

    def forward(self, pred_joints):
        loss_dict = {}
        
        pred_joints = pred_joints[:,self.halpe2lsp]
        
        left_bone_length = torch.norm(pred_joints[:, self.left_start] - pred_joints[:, self.left_end], dim=-1)
        right_bone_length = torch.norm(pred_joints[:, self.right_start] - pred_joints[:, self.right_end], dim=-1)

        skeleton_loss = self.criterion_joint(left_bone_length, right_bone_length).mean()

        loss_dict['skeleton_loss'] = skeleton_loss * self.skeleton_weight
        return loss_dict

class Joint_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, gt_joints, has_3d):
        loss_dict = {}
        
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp]

        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

        gt_pelvis = (gt_joints[:,2,:3] + gt_joints[:,3,:3]) / 2.
        gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

        pred_pelvis = (pred_joints[:,2,:] + pred_joints[:,3,:]) / 2.
        pred_joints = pred_joints - pred_pelvis[:,None,:]

        gt_joints = gt_joints[has_3d == 1]
        pred_joints = pred_joints[has_3d == 1]

        if len(gt_joints) > 0:
            joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['joint_loss'] = joint_loss * self.joint_weight
        return loss_dict


class Plane_Loss(nn.Module):
    def __init__(self, device):
        super(Plane_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.height_weight = 1

    def forward(self, pred_joints, valids):
        loss_dict = {}
        batchsize = len(valids)

        idx = 0
        loss = 0.
        for img in valids.detach().to(torch.int8):
            num = img.sum()

            if num <= 1:
                dis_std = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
            else:
                joints = pred_joints[idx:idx+num]

                bottom = (joints[:,15] + joints[:,16]) / 2
                top = joints[:,17]

                l = (top - bottom) / torch.norm(top - bottom, dim=1)[:,None]
                norm = torch.mean(l, dim=0)

                root = (joints[:,11] + joints[:,12]) / 2 #joints[:,19]

                proj = torch.matmul(root, norm)

                dis_std = proj.std()

            idx += num
            loss += dis_std

        loss_dict['plane_loss'] = loss / batchsize * self.height_weight
        
        return loss_dict

class Joint_reg_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_reg_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0
        # self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, gt_joints, has_3d):
        loss_dict = {}
        
        # pred_joints = pred_joints[:,self.halpe2lsp]
        # gt_joints = gt_joints[:,self.halpe2lsp]

        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

        # gt_pelvis = (gt_joints[:,2,:3] + gt_joints[:,3,:3]) / 2.
        gt_pelvis = gt_joints[:,19,:3]
        gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

        # pred_pelvis = (pred_joints[:,2,:] + pred_joints[:,3,:]) / 2.
        pred_pelvis = pred_joints[:,19,:3]
        pred_joints = pred_joints - pred_pelvis[:,None,:]

        gt_joints = gt_joints[has_3d == 1]
        pred_joints = pred_joints[has_3d == 1]

        if len(gt_joints) > 0:
            joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['Joint_reg_Loss'] = joint_loss * self.joint_weight
        return loss_dict

class Shape_reg(nn.Module):
    def __init__(self, device):
        super(Shape_reg, self).__init__()
        self.device = device
        self.reg_weight = 0.001

    def forward(self, pred_shape):
        loss_dict = {}
        
        loss = torch.norm(pred_shape, dim=1)
        loss = loss.mean()


        loss_dict['shape_reg_loss'] = loss * self.reg_weight
        return loss_dict


class L2(nn.Module):
    def __init__(self, device):
        super(L2, self).__init__()
        self.device = device
        self.L2Loss = nn.MSELoss(size_average=False)

    def forward(self, x, y):
        b = x.shape[0]
        diff = self.L2Loss(x, y)
        diff = diff / b
        return diff



class MPJPE(nn.Module):
    def __init__(self, device):
        super(MPJPE, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff.detach().cpu().numpy()

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}

        # from utils.gui_3d import Gui_3d
        # gui = Gui_3d()

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def pa_mpjpe(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,self.halpe2lsp,-1].detach().cpu()

        pred_joints = pred_joints[:,self.halpe2lsp].detach().cpu()
        gt_joints = gt_joints[:,self.halpe2lsp,:3].detach().cpu()

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:,None,:].repeat(1, 14, 1)

class MPJPE_H36M(nn.Module):
    def __init__(self, device):
        super(MPJPE_H36M, self).__init__()
        self.h36m_regressor = torch.from_numpy(np.load('data/J_regressor_h36m.npy')).to(torch.float32).to(device)
        self.halpe_regressor = torch.from_numpy(np.load('data/J_regressor_halpe.npy')).to(torch.float32).to(device)
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.halpe2h36m = [19,12,14,16,11,13,15,19,19,18,17,5,7,9,6,8,10]
        self.BEV_H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 0]

    def forward_instance(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff.detach().cpu().numpy()

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}

        # from utils.gui_3d import Gui_3d
        # gui = Gui_3d()

        conf = gt_joints[:,:,-1]

        h36m_joints = torch.matmul(self.h36m_regressor, pred_joints)
        halpe_joints = torch.matmul(self.halpe_regressor, pred_joints)

        pred_joints = halpe_joints[:,self.halpe2h36m]
        pred_joints[:,[7,8,9,10]] = h36m_joints[:,[7,8,9,10]]
        gt_joints = gt_joints[:,:,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='h36m')
        gt_joints = self.align_by_pelvis(gt_joints, format='h36m')

        # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='h36m')

        # pred_joints = pred_joints[:,self.BEV_H36M_TO_J14]
        # gt_joints = gt_joints[:,self.BEV_H36M_TO_J14]
        # conf = conf[:,self.BEV_H36M_TO_J14]

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def pa_mpjpe(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,self.halpe2lsp,-1].detach().cpu()

        pred_joints = pred_joints[:,self.halpe2lsp].detach().cpu()
        gt_joints = gt_joints[:,self.halpe2lsp,:3].detach().cpu()

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[:,pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[:,pelvis_id, :]

        return joints - pelvis[:,None,:]

class PCK(nn.Module):
    def __init__(self, device):
        super(PCK, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_joints, gt_joints):
        loss_dict = {}
        confs = gt_joints[:,self.halpe2lsp][:,:,-1]
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp][:,:,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        joint_error = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=-1) * confs)
        diff = torch.mean((joint_error < 0.15).float(), dim=1)
        diff = diff * 100
        
        return diff.detach().cpu().numpy()

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}
        confs = gt_joints[:,self.halpe2lsp][:,:,-1].reshape(-1,)
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp][:,:,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        joint_error = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=-1)).reshape(-1,)
        joint_error = joint_error[confs==1]
        diff = torch.mean((joint_error < 0.15).float(), dim=0)
        diff = diff * 100
        
        return diff

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:,None,:].repeat(1, 14, 1)

