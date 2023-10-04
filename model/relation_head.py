
import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from .HumanGroupNet import MS_HGNN_oridinary,MS_HGNN_hyper
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import matrix_to_axis_angle
import cv2

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])


class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=2048):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)
        self.nmp_layers =args.nmp_layers
            
        # self.input_fc = nn.Linear(in_dim, self.model_dim * 4)
        # self.input_fc2 = nn.Linear(self.model_dim * 4, self.model_dim * 2)
        # self.input_fc3 = nn.Linear(self.model_dim * 2, self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            embedding_dim=2048,
            h_dim=self.model_dim,
            mlp_dim=2048,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=self.nmp_layers
        )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[0]
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[1]
            )

        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[2]
            )

    
    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:5,0] = 1
        category[5:10,1] = 1
        category[10,2] = 1
        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x

    def convert_color(self, gray):
        im_color = cv2.applyColorMap(cv2.convertScaleAbs(gray, alpha=1),cv2.COLORMAP_JET)
        return im_color

    def viz_two_affinity(self, collectives, corrs):
        
        collectives = collectives.detach().cpu().numpy()
        corrs = corrs.detach().cpu().numpy()
        for collective, corr in zip(collectives, corrs):
            ratiox = 800/int(collective.shape[0])
            ratioy = 800/int(collective.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
        
            collective = self.convert_color(collective*255)
            corr = self.convert_color(corr*255)
            # im = cv2.resize(im, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.namedWindow('collective',0)
            cv2.resizeWindow('collective',int(collective.shape[1]*ratio),int(collective.shape[0]*ratio))
            cv2.imshow('collective',collective)
            cv2.namedWindow('affinity',0)
            cv2.resizeWindow('affinity',int(corr.shape[1]*ratio),int(corr.shape[0]*ratio))
            cv2.imshow('affinity',corr)
            cv2.waitKey()

    def viz_affinity(self, aff_map):
        viz = []
        aff_maps = aff_map.detach().cpu().numpy()
        for im in aff_maps:
            ratiox = 800/int(im.shape[0])
            ratioy = 800/int(im.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
        
            im = self.convert_color(im*255)
            # im = cv2.resize(im, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.namedWindow('affinity',0)
            cv2.resizeWindow('affinity',int(im.shape[1]*ratio),int(im.shape[0]*ratio))
            cv2.imshow('affinity',im)
            cv2.waitKey()
            viz.append(im)
        return viz

    def forward(self, inputs, batch_size, agent_num, mask):
        length = inputs.shape[1]

        inputs = inputs * mask[:,None]

        # inputs = self.input_fc(inputs) #.view(batch_size*agent_num, self.model_dim)
        # inputs = self.input_fc2(inputs)
        # inputs = self.input_fc3(inputs)
        # tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*agent_num)
        # tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)
  
        # ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, length*self.model_dim))
        # ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        mask = mask.view(batch_size, agent_num)
        mask = torch.matmul(mask[:,:,None], mask[:,None,:])

        ftraj_input = inputs.view(batch_size, agent_num, -1)

        query_input = F.normalize(ftraj_input,p=2,dim=2)
        feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))

        viz_affinity = False
        if viz_affinity:
            aff_maps = self.viz_affinity(feat_corr)

        ftraj_inter,_ = self.interaction(ftraj_input, mask)

        if len(self.args.hyper_scales) > 0:
            ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 1:
            ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 2:
            ftraj_inter_hyper3,_ = self.interaction_hyper3(ftraj_input,feat_corr, mask)

        if len(self.args.hyper_scales) == 0:
            final_feature = torch.cat((ftraj_input,ftraj_inter),dim=-1)
        if len(self.args.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper),dim=-1)
        elif len(self.args.hyper_scales) == 2:
            final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        elif len(self.args.hyper_scales) == 3:
            final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3),dim=-1)

        output_feature = final_feature.view(batch_size*agent_num,-1)

        return output_feature


        
class relation_head(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)

        # models
        scale_num = 2 + len(self.args.hyper_scales)

        self.past_encoder = PastEncoder(self.args)

        embed_dim = 2048
        out_dim = 24 * 6
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, hidden_dim),
        )
        self.project1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3 , out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3 , 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, 10),
        )

    def set_device(self, device):
        self.device = device
        self.to(device)
    

    def forward(self, data):
        batch_size, agent_num, d = data['features'].shape

        valid = data['valid'].reshape(-1,)

        features = data['features'].reshape(-1, d)

        center = data['center'].reshape(batch_size*agent_num, -1)
        scale = data['scale'].reshape(batch_size*agent_num,)
        img_h = data['img_h'].reshape(batch_size*agent_num,)
        img_w = data['img_w'].reshape(batch_size*agent_num,)
        focal_length = data['focal_length'].reshape(batch_size*agent_num,)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]


        aff_features = torch.cat([features, bbox_info], 1)
        inputs = self.project(aff_features)

        features = self.project1(features)

        relation_features = self.past_encoder(inputs, batch_size, agent_num, valid)

        # extract valid feature
        features = features[valid == 1]
        relation_features = relation_features[valid == 1]
        center = center[valid == 1]
        scale = scale[valid == 1]
        img_h = img_h[valid == 1]
        img_w = img_w[valid == 1]
        focal_length = focal_length[valid == 1]
        bbox_info = bbox_info[valid == 1]

        feature = torch.cat([features, relation_features], dim=1) 

        num_valid = len(feature)

        xc = torch.cat([feature, bbox_info],1)   #may need a fc before self.head
        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc).view(num_valid, 10)
        pred_cam = self.cam_head(xc).view(num_valid, 3)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(num_valid, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(num_valid, 72)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
        
        temp_trans = torch.zeros((pred_rotmat.shape[0], 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)

        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                   rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                   translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
                                                   focal_length=focal_length,
                                                   camera_center=camera_center)
        # pred_keypoints_2d = pred_keypoints_2d / (self.img_res / 2.)

        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256


        pred = {'pred_pose':pred_pose,\
                'pred_shape':pred_shape,\
                'pred_cam_t':pred_trans,\
                'pred_rotmat':pred_rotmat,\
                'pred_verts':pred_verts,\
                'pred_joints':pred_joints,\
                'focal_length':focal_length,\
                'pred_keypoints_2d':pred_keypoints_2d,\
                 }

        return pred

