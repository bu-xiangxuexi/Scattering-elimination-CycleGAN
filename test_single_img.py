import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import cv2
import tifffile

os.environ['CUDA_VISIBLE_DEVICES']='0'
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
# from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch
import skimage

# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="remote_image", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)


writer = SummaryWriter(f'./log/{opt.dataset_name}') # 保存训练过程中的loss，使用tensorboard


# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
# G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = G_BA.cuda()

model_path = './G_BA_100.pth'

#model_path = './model_scatter/model/G_BA_1.pth'
model_path = './model_scatter/model/G_BA_60.pth'

#model_path = './model_scatter/model/G_BA_120.pth'
#model_path = './model_scatter/model/G_BA_180.pth'


if torch.cuda.is_available():
    G_BA.load_state_dict(torch.load(model_path))
else:
    G_BA.load_state_dict(torch.load(model_path,map_location='cpu'))
G_BA.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize((512,512)),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
mean = np.array([0.5,0.5,0.5])
std = np.array([0.5,0.5,0.5])



test_dir = 'test2'
# test_dir = 'test' #路径


def sample_images():
    ssim = 0
    psnr = 0
    """Saves a generated sample from the test set"""
    
    
    res_dir = test_dir+'_res'
    os.makedirs(res_dir, exist_ok=True)
    for i in os.listdir(test_dir):
        print(i)
        tif_path = os.path.join(test_dir,i)
        tif_path_22 = os.path.join(test_dir,i)
        
        # image = cv2.imread(tif_path)
        # image_22 = cv2.imread(tif_path_22)
        
        real_A = Image.open(tif_path)
        real_B = Image.open(tif_path_22)
        print(real_A.size)
        
        
        transform = transforms.Compose(transforms_)
        
        # mask_all  = np.zeros_like(image)
        
        # real_A = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # real_B = cv2.cvtColor(image_22, cv2.COLOR_BGR2RGB)
        
        real_A = transform(real_A)
        real_B = transform(real_B)
        real_A = real_A.cuda()
        real_B = real_B.cuda()
        real_A = real_A.unsqueeze(0)
        real_B = real_B.unsqueeze(0)

        G_BA.eval()
        fake_A = G_BA(real_B)
        
        real_A_arr = real_A.detach().cpu()[0].permute(1,2,0).numpy()
        fake_A_arr = fake_A.detach().cpu()[0].permute(1, 2, 0).numpy()
        real_B_arr = real_B.detach().cpu()[0].permute(1, 2, 0).numpy()
        
        
        real_A = real_A.detach().cpu()[0].permute(1,2,0).numpy()
        real_A = real_A*std + mean 
        real_A = real_A*255.0
        real_A = real_A.astype(np.uint8)
        
        
        real_A = cv2.cvtColor(real_A, cv2.COLOR_BGR2RGB)
        
        real_B = real_B.detach().cpu()[0].permute(1,2,0).numpy()
        real_B_arr = real_B*std + mean 
        real_B_arr = real_B_arr*255.0
        
        fake_A = fake_A.detach().cpu()[0].permute(1,2,0).numpy()
        fake_A = fake_A*std + mean 
        fake_A = fake_A*255.0
        fake_A = fake_A.astype(np.uint8)
        fake_A = cv2.cvtColor(fake_A, cv2.COLOR_BGR2RGB)
        
        # print(fake_A)
        # cv2.imwrite(res_dir+'\\'+i[:-4]+'.png', np.concatenate([real_A, fake_A]))
        # cv2.imwrite(i[:-4]+'.png', np.concatenate([real_A, fake_A]))

        # cv2.imwrite(test_dir+'_'+i[:-4]+'.png', np.concatenate([real_A, fake_A]))
        print(fake_A.shape)
        cv2.imwrite(res_dir+'/'+i[:-4]+'.png', np.concatenate([real_A, fake_A]))

        real_A = cv2.cvtColor(real_A, cv2.COLOR_BGR2RGB)
        fake_A = cv2.cvtColor(fake_A, cv2.COLOR_BGR2RGB)

        
        tifffile.imwrite(res_dir+'/'+i[:-4]+'.tif', np.concatenate([real_A, fake_A]))



        

        # img1 = cv2.imdecode(np.fromfile('./'+test_dir+'_'+i[:-4]+'.png',dtype=np.uint8),-1)
        # cv2.imencode('.png', img1 )[1].tofile(out_path)
# ----------
# testing
# ----------
sample_images()



