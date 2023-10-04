'''
 @FileName    : modules.py
 @EditTime    : 2022-09-27 14:45:21
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import os
import torch
import time
import yaml
from datasets.demo_data import DemoData
from utils.imutils import vis_img
from utils.logger import Logger
from loss_func import *
import torch.optim as optim
from utils.cyclic_scheduler import CyclicLRWithRestarts
from utils.smpl_torch_batch import SMPLModel
from utils.renderer_pyrd import Renderer
import cv2
from thop import profile
from copy import deepcopy
from utils.imutils import joint_projection
from utils.gui_3d import Gui_3d
from utils.FileLoaders import save_pkl
from utils.visualize_pose import show_poses
from utils.pose import Pose

def init(note='occlusion', dtype=torch.float32, mode='eval', **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join('output', note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="template")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    if mode == 'eval':
        logger.set_names(['Surface', 'MPJPE', 'PA-MPJPE', 'PCK'])
    else:
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Test Loss'])

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/SMPL_NEUTRAL.pkl', 
                        data_type=dtype,
                    )

    return out_dir, logger, model_smpl


class LossLoader():
    def __init__(self, train_loss='L1', test_loss='L1', device=torch.device('cpu'), **kwargs):
        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.device = device

        # Parse the loss functions
        self.train_loss = {}
        for loss in self.train_loss_type:
            if loss == 'L1':
                self.train_loss.update(L1=L1(self.device))
            if loss == 'L2':
                self.train_loss.update(L2=L2(self.device))
            if loss == 'SMPL_Loss':
                self.train_loss.update(SMPL_Loss=SMPL_Loss(self.device))
            if loss == 'Keyp_Loss':
                self.train_loss.update(Keyp_Loss=Keyp_Loss(self.device))
            if loss == 'Mesh_Loss':
                self.train_loss.update(Mesh_Loss=Mesh_Loss(self.device))
            if loss == 'Joint_Loss':
                self.train_loss.update(Joint_Loss=Joint_Loss(self.device))
            if loss == 'Skeleton_Loss':
                self.train_loss.update(Skeleton_Loss=Skeleton_Loss(self.device))
            if loss == 'Shape_reg':
                self.train_loss.update(Shape_reg=Shape_reg(self.device))
            if loss == 'Joint_reg_Loss':
                self.train_loss.update(Joint_reg_Loss=Joint_reg_Loss(self.device))
            if loss == 'Plane_Loss':
                self.train_loss.update(Plane_Loss=Plane_Loss(self.device))
            # You can define your loss function in loss_func.py, e.g., Smooth6D, 
            # and load the loss by adding the following lines


        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=L1(self.device))
            if loss == 'MPJPE':
                self.test_loss.update(MPJPE=MPJPE(self.device))
            if loss == 'MPJPE_H36M':
                self.test_loss.update(MPJPE_H36M=MPJPE_H36M(self.device))
            if loss == 'PA_MPJPE':
                self.test_loss.update(PA_MPJPE=MPJPE(self.device))
            if loss == 'MPJPE_instance':
                self.test_loss.update(MPJPE_instance=MPJPE(self.device))
            if loss == 'PCK':
                self.test_loss.update(PCK=PCK(self.device))
            if loss == 'PCK_instance':
                self.test_loss.update(PCK_instance=PCK(self.device))

    def calcul_trainloss(self, pred, gt):
        loss_dict = {}
        gt['has_smpl'] = gt['has_smpl'].squeeze(1)
        gt['has_3d'] = gt['has_3d'].squeeze(1)

        for ltype in self.train_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.train_loss['L1'](pred, gt))
            elif ltype == 'L2':
                loss_dict.update(L2=self.train_loss['L2'](pred, gt))
            elif ltype == 'SMPL_Loss':
                SMPL_loss = self.train_loss['SMPL_Loss'](pred['pred_rotmat'], gt['pose'], pred['pred_shape'], gt['betas'], gt['has_smpl'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Plane_Loss':
                Plane_Loss = self.train_loss['Plane_Loss'](pred['pred_joints'], gt['valid'])
                loss_dict = {**loss_dict, **Plane_Loss}
            elif ltype == 'Keyp_Loss':
                Keyp_loss = self.train_loss['Keyp_Loss'](pred['pred_keypoints_2d'], gt['keypoints'])
                loss_dict = {**loss_dict, **Keyp_loss}
            elif ltype == 'Mesh_Loss':
                Mesh_loss = self.train_loss['Mesh_Loss'](pred['pred_verts'], gt['verts'], gt['has_smpl'])
                loss_dict = {**loss_dict, **Mesh_loss}
            elif ltype == 'Joint_Loss':
                Joint_Loss = self.train_loss['Joint_Loss'](pred['pred_joints'], gt['gt_joints'], gt['has_3d'])
                loss_dict = {**loss_dict, **Joint_Loss}
            elif ltype == 'Skeleton_Loss':
                Skeleton_Loss = self.train_loss['Skeleton_Loss'](pred['pred_joints'])
                loss_dict = {**loss_dict, **Skeleton_Loss}
            elif ltype == 'Joint_abs_Loss':
                pred_joints_abs = pred['pred_joints'] + pred['pred_cam_t'][:,None,:]
                gt_joints_abs = gt['gt_joints'].detach()
                gt_joints_abs[:,:,:3] = gt_joints_abs[:,:,:3] + gt['gt_cam_t'][:,None,:]
                Joint_abs_Loss = self.train_loss['Joint_abs_Loss'](pred_joints_abs, gt_joints_abs, gt['has_3d'])
                loss_dict = {**loss_dict, **Joint_abs_Loss}
            elif ltype == 'Shape_reg':
                Shape_reg = self.train_loss['Shape_reg'](pred['pred_shape'])
                loss_dict = {**loss_dict, **Shape_reg}
            elif ltype == 'Pose_reg':
                Pose_reg = self.train_loss['Pose_reg'](pred['pred_pose'])
                loss_dict = {**loss_dict, **Pose_reg}
            elif ltype == 'Joint_reg_Loss':
                Joint_reg_Loss = self.train_loss['Joint_reg_Loss'](pred['transformer_joints'], gt['gt_joints'], gt['has_3d'])
                loss_dict = {**loss_dict, **Joint_reg_Loss}
            # Calculate your loss here

            # elif ltype == 'Smooth6D':
            #     loss_dict.update(Smooth6D=self.train_loss['Smooth6D'](pred_pose))
            else:
                pass
        loss = 0
        for k in loss_dict:
            loss_temp = loss_dict[k] * 60.
            loss += loss_temp
            loss_dict[k] = round(float(loss_temp.detach().cpu().numpy()), 6)
        return loss, loss_dict


    def calcul_testloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred, gt))
            elif ltype == 'MPJPE':
                loss_dict.update(MPJPE=self.test_loss['MPJPE'](pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'MPJPE_H36M':
                loss_dict.update(MPJPE_H36M=self.test_loss['MPJPE_H36M'](pred['pred_verts'], gt['gt_joints']))
            elif ltype == 'PA_MPJPE':
                loss_dict.update(PA_MPJPE=self.test_loss['PA_MPJPE'].pa_mpjpe(pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'PCK':
                loss_dict.update(PCK=self.test_loss['PCK'](pred['pred_joints'], gt['gt_joints']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass
        loss = 0
        for k in loss_dict:
            loss += loss_dict[k]
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 6)
        return loss, loss_dict

    def calcul_instanceloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred, gt))
            elif ltype == 'MPJPE_instance':
                loss_dict.update(MPJPE_instance=self.test_loss['MPJPE_instance'].forward_instance(pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'PCK_instance':
                loss_dict.update(PCK_instance=self.test_loss['PCK_instance'].forward_instance(pred['pred_joints'], gt['gt_joints']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass

        return loss_dict

def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    data = {'features':torch.zeros((1,8,2048), device=next(model.parameters()).device),
            'center':torch.zeros((8,2), device=next(model.parameters()).device),
            'scale':torch.zeros((8,), device=next(model.parameters()).device),
            'valid':torch.ones((8,), device=next(model.parameters()).device),
            'img_h':torch.zeros((8,), device=next(model.parameters()).device),
            'img_w':torch.zeros((8,), device=next(model.parameters()).device),
            'focal_length':torch.zeros((8,), device=next(model.parameters()).device)}
    flops, params = profile(deepcopy(model), inputs=(data,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= 2  # Gflops
    # flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

class ModelLoader():
    def __init__(self, dtype=torch.float32, output='', device=torch.device('cpu'), model=None, lr=0.001, pretrain=False, pretrain_dir='', batchsize=32, task=None, data_folder='', use_prior=False, testset='', test_loss='MPJPE', **kwargs):

        self.output = output
        self.device = device
        self.batchsize = batchsize
        self.data_folder = data_folder
        self.test_loss = test_loss
        if self.test_loss in ['PCK']:
            self.best_loss = -1
        else:
            self.best_loss = 999999999

        # load smpl model 
        self.model_smpl_gpu = SMPLModel(
                            device=torch.device('cuda'),
                            model_path='./data/SMPL_NEUTRAL.pkl', 
                            data_type=dtype,
                        )
        # # Setup renderer for visualization
        # self.renderer = Renderer(focal_length=5000., img_res=224, faces=self.model_smpl_gpu.faces)

        if testset == 'JTA':
            num_joint = 15
        else:
            num_joint = 21

        # Load model according to model name
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(self.model_smpl_gpu, num_joints=num_joint)
        print('load model: %s' %self.model_type)

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count: %.2fM' % (model_params / 1e6))


        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        # print("Model Summary: {}".format(get_model_info(self.model, (800, 1440))))

        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = None

        # Load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print("Load pretrain parameters from %s" %pretrain_dir)
            self.optimizer.load_state_dict(params['optimizer'])
            print("Load optimizer parameters")
            

        if task == 'relation' and use_prior:
            model_dict = self.model.state_dict()
            params = torch.load('pretrain_model/mix_hmr300.pkl')
            premodel_dict = params['model']
            premodel_dict = {'backbone.' + k: v for k ,v in premodel_dict.items() if 'backbone.' + k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)

            # for parameter in self.model.backbone.parameters():
            #     parameter.requires_grad == False

    def load_scheduler(self, epoch_size):
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=epoch_size, restart_period=10, t_mult=2, policy="cosine", verbose=True)

    def load_checkpoint(self, pretrain_dir):
        model_dict = self.model.state_dict()
        params = torch.load(pretrain_dir)
        premodel_dict = params['model']
        premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        model_dict.update(premodel_dict)
        self.model.load_state_dict(model_dict)
        print("Load pretrain parameters from %s" %pretrain_dir)

    def save_model(self, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, '%s_epoch%03d.pkl' %(task, epoch))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save model to %s' % model_name)

    def save_best_model(self, testing_loss, epoch, task):
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        if self.test_loss in ['PCK']:
            if self.best_loss < testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)
        else:
            if self.best_loss > testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)

    def save_camparam(self, path, intris, extris):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        f = open(path, 'w')
        for ind, (intri, extri) in enumerate(zip(intris, extris)):
            f.write(str(ind)+'\n')
            for i in intri:
                f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
            f.write('0 0 \n')
            for i in extri[:3]:
                f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
            f.write('\n')
        f.close()

    def save_params(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        for index, (img, pred_trans, pred_pose, pred_shape, h, w) in enumerate(zip(results['imgs'], results['pred_trans'], results['pred_pose'], results['pred_shape'], results['img_h'], results['img_w'])):
            name = img.replace(self.data_folder + '\\', '').replace('.jpg', '')
            data = {}
            data['pose'] = pred_pose
            data['trans'] = pred_trans
            data['betas'] = pred_shape

            intri = np.eye(3)
            intri[0][0] = (w**2 + h**2)**0.5
            intri[1][1] = (w**2 + h**2)**0.5
            intri[0][2] = w / 2
            intri[1][2] = h / 2
            extri = np.eye(4)
            
            cam_path = os.path.join(self.output, 'camparams', name)
            os.makedirs(cam_path, exist_ok=True)
            self.save_camparam(os.path.join(cam_path, 'camparams.txt'), [intri], [extri])

            path = os.path.join(self.output,  name)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, '%04d.pkl' %index)
            save_pkl(path, data)

    def save_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_verts'] = results['gt_verts'] + results['gt_trans'][:,np.newaxis,:]

        for index, (img, pred_verts, gt_verts, focal) in enumerate(zip(results['imgs'], results['pred_verts'], results['gt_verts'], results['focal_length'])):
            # print(img)
            name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            pred_smpl = renderer.render_front_view(pred_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())

            # gt_smpl = renderer.render_front_view(gt_verts[np.newaxis,:,:],
            #                                         bg_img_rgb=img.copy())

            render_name = "%s_%02d_pred_smpl.jpg" % (name, iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), pred_smpl)

            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            mesh_name = os.path.join(output, 'meshes/%s_%02d_pred_mesh.obj' %(name, iter * batchsize + index))
            self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_gt_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

    def save_joint_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        
        results['pred_joints'] = results['pred_joints'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_joints'] = results['gt_joints'][:,:,:3] + results['gt_trans'][:,np.newaxis,:]

        # gui_3d = Gui_3d()
        # gui_3d.vis_skeleton(results['pred_joints'][:,5:19], results['gt_joints'][:,5:19])


        for index, (img, pred_joints, gt_joints, focal) in enumerate(zip(results['imgs'], results['pred_joints'], results['gt_joints'], results['focal_length'])):
            # print(img)
            name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = img_w / 2
            intri[1][2] = img_h / 2

            pred_joints, _ = joint_projection(pred_joints[5:19], np.eye(4), intri, img, viz=False)
            gt_joints, _ = joint_projection(gt_joints[5:19], np.eye(4), intri, img, viz=False)

            for p in gt_joints.astype(np.int):
                img = cv2.circle(img, tuple(p), 5, (0,0,255), -1)

            for p in pred_joints.astype(np.int):
                img = cv2.circle(img, tuple(p), 4, (0,255,255), -1)

            render_name = "%s_%02d_pred_joint.jpg" % (name, iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), img)

            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

    def save_demo_joint_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_joints'] = results['pred_joints'] + results['pred_trans'][:,np.newaxis,:]

        name = results['imgs'].replace(self.data_folder + '\\', '').replace(self.data_folder + '/', '').replace('\\', '_').replace('/', '_')
        img = cv2.imread(results['imgs'])

        for index, (pred_joints, focal) in enumerate(zip(results['pred_joints'], results['focal_length'])):

            img_h, img_w = img.shape[:2]
            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = img_w / 2
            intri[1][2] = img_h / 2

            proj_joints, _ = joint_projection(pred_joints[5:19], np.eye(4), intri, img, viz=False)
            proj_joints = proj_joints.astype(np.int)

            for c, limb in enumerate(Pose.LIMBS_HALPE_14):
                img = cv2.line(img, tuple(proj_joints[limb[0]]), tuple(proj_joints[limb[1]]), (0,255,255), 3)

            for p in proj_joints.astype(np.int):
                img = cv2.circle(img, tuple(p), 4, (0,0,255), -1)

        render_name = "%s_%02d_pred_joint.jpg" % (name, iter * batchsize + index)
        cv2.imwrite(os.path.join(output, render_name), img)

        vis_img('projection', img)
        show_poses(results['pred_joints'][:,5:19])





    def save_hmr_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_verts'] = results['gt_verts'] + results['gt_trans'][:,np.newaxis,:]
        results['input_img'] = results['input_img'].transpose((0,2,3,1))[...,::-1]

        for index, (img, input_img, pred_verts, gt_verts) in enumerate(zip(results['imgs'], results['input_img'], results['pred_verts'], results['gt_verts'])):
            # print(img)
            name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')
            img = (input_img*255.).astype(np.uint8)
            focal = 5000.
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            pred_smpl = renderer.render_front_view(pred_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())

            gt_smpl = renderer.render_front_view(gt_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())


            render_name = "%s_pred_smpl.jpg" % (name)
            cv2.imwrite(os.path.join(output, render_name), pred_smpl)

            render_name = "%s_gt_smpl.jpg" % (name)
            cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            mesh_name = os.path.join(output, 'meshes/%s_pred_mesh.obj' %(name))
            self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            mesh_name = os.path.join(output, 'meshes/%s_gt_mesh.obj' %(name))
            self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)



    def save_demo_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]

        for index, (img, pred_verts, focal, input) in enumerate(zip(results['imgs'], results['pred_verts'], results['focal_length'], results['origin_input'])):
            # print(img)
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            pred_smpl = renderer.render_front_view(pred_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())

            for kp in input:
                pred_smpl = cv2.circle(pred_smpl, tuple(kp[:2].astype(np.int)), 5, (0,0,255), -1)

            render_name = "%05d_pred_smpl.jpg" % (iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), pred_smpl)

            mesh_name = os.path.join(output, 'meshes/%05d_pred_mesh.obj' %(iter * batchsize + index))
            self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

class DatasetLoader():
    def __init__(self, trainset=None, testset=None, data_folder='./data', dtype=torch.float32, smpl=None, task=None, model='hmr', **kwargs):
        self.data_folder = data_folder
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.dtype = dtype
        self.smpl = smpl
        self.task = task
        self.model = model

    def load_demo_data(self):
        test_dataset = DemoData(False, self.dtype, self.data_folder, '', self.smpl)

        return test_dataset
