'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import numpy as np
import cv2
from tqdm import tqdm
import time

def extract_valid(data):
    batch_size, agent_num, d = data['keypoints'].shape[:3]
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'] #.reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'] #.reshape(batch_size*agent_num,)[valid == 1]

    data['valid_img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]
    data['has_3d'] = data['has_3d'].reshape(batch_size*agent_num,1)[valid == 1]
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*agent_num,1)[valid == 1]
    data['verts'] = data['verts'].reshape(batch_size*agent_num, 6890, 3)[valid == 1]
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*agent_num, -1, 4)[valid == 1]
    data['pose'] = data['pose'].reshape(batch_size*agent_num, 72)[valid == 1]
    data['betas'] = data['betas'].reshape(batch_size*agent_num, 10)[valid == 1]
    data['keypoints'] = data['keypoints'].reshape(batch_size*agent_num, 26, 3)[valid == 1]
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*agent_num, 3)[valid == 1]

    data['ori_imgname'] = data['imgname']
    imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['img'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center']
    data['scale'] = data['scale']
    data['img_h'] = data['img_h']
    data['img_w'] = data['img_w']
    data['focal_length'] = data['focal_length']

    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}

    return data

def relation_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        data = to_device(data, device)
        data = extract_valid(data)

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach() #/ batchsize
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        train_loss += loss_batch

    return train_loss/len_data

def relation_test(model, loss_func, loader, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['keypoints'].shape[0]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if False:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['valid_img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['valid_img_w'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if i < 1:
                results = {}
                results.update(imgs=data['ori_imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['valid_focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(valid=data['valid'].detach().cpu().numpy().astype(np.float32))

                if 'MPJPE_instance' in cur_loss_dict.keys() or 'MPJPE_H36M_instance' in cur_loss_dict.keys():
                    results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))

                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_test_results(results, i, batchsize)

            loss_batch = loss.detach().mean() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def relation_demo(model, loader, device=torch.device('cpu')):

    print('-' * 10 + 'model demo' + '-' * 10)
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].shape[0]
            data = to_device(data, device)
            data = extract_valid_demo(data)

            # forward
            pred = model.model(data)

            results = {}
            results.update(imgs=data['imgname'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['valid_focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_results(results, i, batchsize)

