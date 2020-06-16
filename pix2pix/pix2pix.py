import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
# import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
# from datasets import *
from mydataset import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.backends import cudnn

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="RIRE-ct-t1", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=17, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
opt = parser.parse_args()
# print(opt)

# os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
# os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

# Loss functions
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixelwise = torch.nn.L1Loss().to(device)

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=1, out_channels=1).to(device)
discriminator = Discriminator(in_channels=1).to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

dataset = MRI_T1_CT_Dataset("../Processed_Data/%s" % opt.dataset_name)
# print(len(dataset))
test_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset:
	np.random.seed(random_seed)
	np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=test_sampler)

anloss_G = []
anloss_D = []

t_loss_G = []
t_loss_D = []

# ----------
#  Training
# ----------
cudnn.benchmark = True
prev_time = time.time()
# fig = plt.figure()
with open("out.csv", "wt") as f:
	for epoch in tqdm(range(opt.epoch, opt.n_epochs)):
		for i, batch in enumerate(tqdm(train_loader)):
			# Model inputs
			real_A = Variable(batch["A"].type(torch.FloatTensor)).to(device)
			real_B = Variable(batch["B"].type(torch.FloatTensor)).to(device)

			# Adversarial ground truths
			valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))), requires_grad=False).to(device)
			fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False).to(device)

			# ------------------
			#  Train Generators
			# ------------------

			optimizer_G.zero_grad()
			# GAN loss
			fake_B = generator(real_A)
			pred_fake = discriminator(fake_B, real_A)
			loss_GAN = criterion_GAN(pred_fake, valid)
			# Pixel-wise loss
			loss_pixel = criterion_pixelwise(fake_B, real_B)
			# Total loss
			loss_G = loss_GAN + lambda_pixel * loss_pixel
			t_loss_G.append(loss_G.item())
			loss_G.backward()
			optimizer_G.step()

			# ---------------------
			#  Train Discriminator
			# ---------------------

			optimizer_D.zero_grad()
			# Real loss
			pred_real = discriminator(real_B, real_A)
			loss_real = criterion_GAN(pred_real, valid)
			# Fake loss
			pred_fake = discriminator(fake_B.detach(), real_A)
			loss_fake = criterion_GAN(pred_fake, fake)
			# Total loss
			loss_D = (0.5 * (loss_real + loss_fake))
			t_loss_D.append(loss_D.item())
			loss_D.backward()
			optimizer_D.step()

	    # test
	    # avg_psnr = 0
	    # for batch in testing_data_loader:
	    #     input, target = batch[0].to(device), batch[1].to(device)

	    #     prediction = net_g(input)
	    #     mse = criterionMSE(prediction, target)
	    #     psnr = 10 * log10(1 / mse.item())
	    #     avg_psnr += psnr
	    # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

			# --------------
			#  Log Progress
			# --------------

			# Determine approximate time left
			batches_done = epoch * len(train_loader) + i
			batches_left = opt.n_epochs * len(train_loader) - batches_done
			time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
			prev_time = time.time()

			# Print log
			# sys.stdout.write(
			# 	"\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
			# 	% (
			# 		epoch,
			# 		opt.n_epochs,
			# 		i,
			# 		len(train_loader),
			# 		loss_D.item(),
			# 		loss_G.item(),
			# 		loss_pixel.item(),
			# 		loss_GAN.item(),
			# 		time_left,
			# 	)
			# )

			f.write(f"{epoch}, {opt.n_epochs}, {i}, {len(train_loader)}, {round(loss_D.item(), 4)}, {round(loss_G.item(), 4)}\n")

			# If at sample interval save image
			# if batches_done % opt.sample_interval == 0:
			# 	sample_images(batches_done)
		anloss_D.append(np.asarray(t_loss_D).mean())
		anloss_G.append(np.asarray(t_loss_G).mean())

		t_loss_D = []
		t_loss_G = []
		# plt.plot(anloss_D, '')
		# plt.plot(anloss_G)
		# Save figure (dpi 300 is good when saving so graph has high resolution)
		# plt.savefig('mygraph.png', dpi=300)

		# if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
		#     # Save model checkpoints
		#     torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
		#     torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
f.close()
