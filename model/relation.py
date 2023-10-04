
import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from model.relation_head import relation_head
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import matrix_to_axis_angle
from model.backbones.resnet import ResNet
import cv2

class relation(relation_head):
    def __init__(self, smpl, num_joints=21):
        super(relation, self).__init__(smpl, num_joints=21)

        self.backbone = ResNet(layers=[3, 4, 6, 3])
        self.backbone.avgpool = nn.AvgPool2d((7, 7), stride=1)

    def forward(self, data):
        batch_size, agent_num = data['img'].shape[:2]
        valid = data['valid'].reshape(-1,)
        
        features = torch.zeros((batch_size*agent_num, 2048), device=data['img'].device, dtype=data['img'].dtype)
        valid_imgs = data['img'].reshape(batch_size * agent_num, 3, 224, 224)[valid == 1]

        features[valid == 1] = self.backbone(valid_imgs)

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

        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256 #constants.IMG_RES


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

