'''
 @FileName    : demo.py
 @EditTime    : 2023-09-28 16:53:59
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''
import torch
from cmd_parser import parse_config
from modules import init, ModelLoader, DatasetLoader

###########Load config file in debug mode#########
import sys
sys.argv = ['','--config=cfg_files/demo.yaml']

def main(**args):

    # Global setting
    dtype = torch.float32
    device = torch.device(index=args.get('gpu_index'), type='cuda')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    eval_dataset = dataset.load_demo_data()

    eval_dataset.human_detection()

    # Load handle function with the task name
    task = args.get('task')
    exec('from process import %s_demo' %task)
    
    eval('%s_demo' %task)(model, eval_dataset, device=device)



if __name__ == "__main__":
    args = parse_config()
    main(**args)








