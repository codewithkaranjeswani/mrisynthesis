import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from util.ssim import ssim
from util.psnr import psnr

from models.AnamNet import AnamNetGenerator
from models.UNet import GeneratorUNet, PatchDiscriminator, weights_init_normal
from models.ENet import ENet
from models.VGGNet import VGG16
from dataset.mrict import MRI_T1_CT_Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.backends import cudnn

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="RIRE-ct-t1", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="epochs after which we sample of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="epochs between model checkpoints")
parser.add_argument("--gen", type=str, default="ENet", help="Selecting generator: UNet | AnamNet | ENet")
parser.add_argument("--in_ch", type=int, default=3, help="Considering neighbouring slices from input")
opt = parser.parse_args()
# print(opt)

os.makedirs("output/", exist_ok=True)
os.makedirs("output/sample_images/", exist_ok=True)
os.makedirs("saved_models/", exist_ok=True)

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
lambda_pixel = 1
lambda_vgg = 0.01

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# specific indices to keep track of during training
spec_ind = [162, 235]

# Initialize generator and discriminator
if (opt.gen == 'UNet'):
	generator = GeneratorUNet(in_channels=opt.in_ch, out_channels=1).to(device)
elif (opt.gen == 'AnamNet'):
	generator = AnamNetGenerator(C=opt.in_ch).to(device)
elif (opt.gen == 'ENet'):
	generator = ENet(C=opt.in_ch).to(device)
else:
	print("Generator not defined.")
	exit(1)
discriminator = PatchDiscriminator(in_channels=opt.in_ch+1).to(device)
vgg = VGG16().to(device)

if opt.epoch != 0:
	# Load pretrained models
	generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
	discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
	# Initialize weights
	generator.apply(weights_init_normal)
	discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def save_trio(realA, fakeB, realB, epoch, i, label, gen):
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	realA = realA[0,1,:,:].T.detach().cpu()
	realA = realA.numpy()
	fakeB = fakeB[0,0,:,:].T.detach().cpu()
	fakeB = fakeB.numpy()
	realB = realB[0,0,:,:].T.detach().cpu()
	realB = realB.numpy()

	ax1.imshow(realA, cmap="gray", origin="lower")
	ax1.set_title("MRI")
	ax2.imshow(fakeB, cmap="gray", origin="lower")
	ax2.set_title("SynCT")
	ax3.imshow(realB, cmap="gray", origin="lower")
	ax3.set_title("CT")
	plt.tight_layout()
	# plt.show()
	plt.savefig('output/sample_images/sample_{0}_{1}_{2}_{3}.png'.format(gen, epoch, label, i), dpi=300)
	plt.close()

def sample_images(epoch):
	for i in range(1):
		# from train data
		ind = torch.randint(0, len(train_indices), size=(1,)).item()
		ind = train_indices[ind]
		sample = dataset[ind]
		realA = sample['A'].to(device)
		realA = realA.view( tuple([1] + list(realA.size())) )
		realB = sample['B'].to(device)
		realB = realB.view( tuple([1] + list(realB.size())) )
		fakeB = generator(realA)
		save_trio(realA, fakeB, realB, epoch+1, i+1, "train", opt.gen)
		ssim_val = ssim(fakeB, realB, data_range=1.0, size_average=False).item()
		psnr_val = psnr(fakeB, realB, data_range=1.0).item()
		s.write(f"{epoch+1},train,{i+1},{round(psnr_val, 4)},{round(ssim_val, 4)}\n")
		# from test data
		ind = torch.randint(0, len(test_indices), size=(1,)).item()
		ind = test_indices[ind]
		sample = dataset[ind]
		realA = sample['A'].to(device)
		realA = realA.view( tuple([1] + list(realA.size())) )
		realB = sample['B'].to(device)
		realB = realB.view( tuple([1] + list(realB.size())) )
		fakeB = generator(realA)
		save_trio(realA, fakeB, realB, epoch+1, i+1, "test", opt.gen)
		ssim_val = ssim(fakeB, realB, data_range=1.0, size_average=False).item()
		psnr_val = psnr(fakeB, realB, data_range=1.0).item()
		s.write(f"{epoch+1},test,{i+1},{round(psnr_val, 4)},{round(ssim_val, 4)}\n")

def sample_special_images(epoch):
	for i in spec_ind:
		sample = dataset[i]
		realA = sample['A'].to(device)
		realA = realA.view( tuple([1] + list(realA.size())) )
		realB = sample['B'].to(device)
		realB = realB.view( tuple([1] + list(realB.size())) )
		fakeB = generator(realA)
		save_trio(realA, fakeB, realB, epoch+1, i+1, "spec", opt.gen)
		ssim_val = ssim(fakeB, realB, data_range=1.0, size_average=False).item()
		psnr_val = psnr(fakeB, realB, data_range=1.0).item()
		m.write(f"{epoch+1},spec,{i+1},{round(psnr_val, 4)},{round(ssim_val, 4)}\n")

cudnn.benchmark = True

dataset = MRI_T1_CT_Dataset("../Processed_Data/%s" % opt.dataset_name)
shuffle_dataset = False
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
# splitting at 13 patients data # 13*18 = 234 # 4*18 = 72 # tot = 17*18 = 306
split = 72
for ki in range(4):
	print("K = {}".format(ki+1))
	start_split = ki * split
	end_split = start_split + split
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, test_indices = indices[:start_split] + indices[end_split:], indices[start_split:end_split]

	# Creating data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	test_sampler = SubsetRandomSampler(test_indices)

	train_loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler)
	test_loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=test_sampler)

	f = open("output/k{0}_loss_pix2pix_{1}_{2}_train.csv".format(ki, opt.gen, opt.n_epochs), "wt")
	g = open("output/k{0}_loss_pix2pix_{1}_{2}_test.csv".format(ki, opt.gen, opt.n_epochs), "wt")
	s = open("output/k{0}_randouts_acc_pix2pix_{1}_{2}.csv".format(ki, opt.gen, opt.n_epochs), "wt")
	m = open("output/k{0}_specouts_acc_pix2pix_{1}_{2}.csv".format(ki, opt.gen, opt.n_epochs), "wt")

	f.write("epoch,D loss,G loss,psnr avg,ssim avg,D pred\n")
	g.write("epoch,D loss,G loss,psnr avg,ssim avg,D pred\n")
	s.write("epoch,type,ind,psnr,ssim\n")
	m.write("epoch,type,ind,psnr,ssim\n")

	# Re-Initialize weights
	generator.apply(weights_init_normal)
	discriminator.apply(weights_init_normal)

	for epoch in tqdm(range(opt.epoch, opt.n_epochs)):
		# ----------
		#  Training
		# ----------
		t_loss_G = []
		t_loss_D = []
		t_ssim = []
		t_psnr = []
		t_pred_d = []

		for i, batch in enumerate(train_loader):
			# Model inputs
			real_A = Variable(batch["A"].type(torch.FloatTensor)).to(device)
			real_B = Variable(batch["B"].type(torch.FloatTensor)).to(device)

			# Adversarial ground truths
			valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))), \
				requires_grad=False).to(device)
			fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))), \
				requires_grad=False).to(device)

			# ------------------
			#  Train Generator
			# ------------------

			optimizer_G.zero_grad()
			# GAN loss
			fake_B = generator(real_A)
			pred_fake = discriminator(fake_B, real_A)
			t_pred_d.append(torch.mean(pred_fake).item())
			loss_GAN = criterion_GAN(pred_fake, valid)
			# Pixel-wise loss
			loss_pixel = criterion_pixelwise(fake_B, real_B)
			# VGG loss
			VGG_real=vgg(real_B.expand([int(real_B.size()[0]),3,int(real_B.size()[2]),int(real_B.size()[3])]))[0]
			VGG_fake=vgg(fake_B.expand([int(real_B.size()[0]),3,int(real_B.size()[2]),int(real_B.size()[3])]))[0]
			VGG_loss=criterion_pixelwise(VGG_fake,VGG_real)
			# Total loss
			loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_vgg * VGG_loss
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

			# metric values
			ssim_val = ssim(fake_B, real_B, data_range=1.0, size_average=False)
			ssim_val = torch.mean(ssim_val).item()
			t_ssim.append(ssim_val)
			psnr_val = psnr(fake_B, real_B, data_range=1.0)
			psnr_val = torch.mean(psnr_val).item()
			t_psnr.append(psnr_val)

			# --------------
			#  Log Progress
			# --------------

			# Determine approximate time left
			# batches_done = epoch * len(train_loader) + i
			# batches_left = opt.n_epochs * len(train_loader) - batches_done
			# time_left = datetime.timedelta(seconds=batches_left * \
			# 	(time.time() - prev_time))
			# prev_time = time.time()

			# f.write(f"{epoch}, {opt.n_epochs}, {i}, {len(train_loader)}, \
			# 	{round(loss_D.item(), 4)}, {round(loss_G.item(), 4)}, \
			# 	{round(psnr_val, 4)}, {round(ssim_val, 4)}, \
			# 	{round(torch.mean(pred_fake), 4)}\n")

		ep_loss_d = np.asarray(t_loss_D).mean()
		ep_loss_g = np.asarray(t_loss_G).mean()
		ep_ssim = np.asarray(t_ssim).mean()
		ep_psnr = np.asarray(t_psnr).mean()
		ep_pred_d = np.asarray(t_pred_d).mean()

		f.write(f"{epoch+1},{round(ep_loss_d, 4)},{round(ep_loss_g, 4)},\
			{round(ep_psnr, 4)},{round(ep_ssim, 4)},{round(ep_pred_d, 4)}\n")

		# ----------
		#  Testing
		# ----------
		t_loss_D = []
		t_loss_G = []
		t_psnr   = []
		t_ssim   = []
		t_pred_d = []
		
		for i, batch in enumerate(test_loader):
			# Model inputs
			real_A = Variable(batch["A"].type(torch.FloatTensor)).to(device)
			real_B = Variable(batch["B"].type(torch.FloatTensor)).to(device)

			# Adversarial ground truths
			valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))), \
				requires_grad=False).to(device)
			fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))), \
				requires_grad=False).to(device)

			with torch.no_grad():

				# ------------------
				#  Test Generator
				# ------------------

				# GAN loss
				fake_B = generator(real_A)
				pred_fake = discriminator(fake_B, real_A)
				t_pred_d.append(torch.mean(pred_fake).item())
				loss_GAN = criterion_GAN(pred_fake, valid)
				# Pixel-wise loss
				loss_pixel = criterion_pixelwise(fake_B, real_B)
				# Total loss
				loss_G = loss_GAN + lambda_pixel * loss_pixel
				t_loss_G.append(loss_G.item())

				# ---------------------
				#  Test Discriminator
				# ---------------------

				# Real loss
				pred_real = discriminator(real_B, real_A)
				loss_real = criterion_GAN(pred_real, valid)
				# Fake loss
				pred_fake = discriminator(fake_B.detach(), real_A)
				loss_fake = criterion_GAN(pred_fake, fake)
				# Total loss
				loss_D = (0.5 * (loss_real + loss_fake))
				t_loss_D.append(loss_D.item())

			# metric values
			ssim_val = ssim(fake_B, real_B, data_range=1.0, size_average=False)
			ssim_val = torch.mean(ssim_val).item()
			t_ssim.append(ssim_val)
			psnr_val = psnr(fake_B, real_B, data_range=1.0)
			psnr_val = torch.mean(psnr_val).item()
			t_psnr.append(psnr_val)

			# g.write(f"{epoch}, {opt.n_epochs}, {i}, {len(train_loader)}, \
			# 	{round(loss_D.item(), 4)}, {round(loss_G.item(), 4)}, \
			# 	{round(psnr_val, 4)}, {round(ssim_val, 4)}, \
			# 	{round(torch.mean(pred_fake), 4)}\n")
		
		ep_loss_d = np.asarray(t_loss_D).mean()
		ep_loss_g = np.asarray(t_loss_G).mean()
		ep_ssim = np.asarray(t_ssim).mean()
		ep_psnr = np.asarray(t_psnr).mean()
		ep_pred_d = np.asarray(t_pred_d).mean()

		g.write(f"{epoch+1},{round(ep_loss_d, 4)},{round(ep_loss_g, 4)},\
			{round(ep_psnr, 4)},{round(ep_ssim, 4)},{round(ep_pred_d, 4)}\n")

		if (epoch+1) % opt.sample_interval == 0:
			sample_images(epoch)
		if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
			# Save model checkpoints
			torch.save(generator.state_dict(), "saved_models/k%d_generator_%d.pth" % (ki, epoch))
			torch.save(discriminator.state_dict(), "saved_models/k%d_discriminator_%d.pth" % (ki, epoch))

	f.close()
	g.close()
	s.close()
	m.close()

