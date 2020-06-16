import random
import os
import numpy as np
import nibabel as nib
import glob

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MRI_T1_CT_Dataset(Dataset):
	def __init__(self, root):
		self.transform = transforms.ToTensor()
		data_dir = os.listdir(root)
		ct_fn = sorted(os.listdir(root + '/' + data_dir[0]))
		mr_fn = sorted(os.listdir(root + '/' + data_dir[1]))
		# print(ct_fn)
		# print(mr_fn)

		# ct_fn = sorted(glob.glob(root + '/' + data_dir[0]))
		# mr_fn = sorted(glob.glob(root + '/' + data_dir[1]))
		# print(ct_fn)
		# print(mr_fn)

		# exit(2)
		self.datalist = []
		for i, (mr_file, ct_file) in enumerate(zip(mr_fn, ct_fn)):
			assert mr_file[:3] == ct_file[:3], "MRI and CT files not from the same patient"
			if (mr_file[:3] == ct_file[:3]):
				ct_img = nib.load(root + '/' + data_dir[0] + '/' + ct_file)
				ct_img_data = ct_img.get_fdata()
				mid = (int)(ct_img_data.shape[2] / 2)
				ct_img_data = ct_img_data[0::2,0::2,mid-10:mid+10]
				ct_img_data = np.array(ct_img_data)
				mr_img = nib.load(root + '/' + data_dir[1] + '/' + mr_file)
				mr_img_data = mr_img.get_fdata()
				mid = (int)(mr_img_data.shape[2] / 2)
				mr_img_data = mr_img_data[:,:,mid-10:mid+10]
				mr_img_data = np.array(mr_img_data)
				assert mr_img_data.shape == ct_img_data.shape, "MRI and CT have different shapes"
				for j in range(20):
					self.datalist.append([ np.uint8(mr_img_data[:,:,j]), np.uint8(ct_img_data[:,:,j]) ])
				# print(mr_file, mr_img_data.shape, ct_file, ct_img_data.shape)

	def __getitem__(self, index):
		[mr_img, ct_img] = self.datalist[index]
		img_A = self.transform(mr_img)
		img_B = self.transform(ct_img)
		return {"A": img_A, "B": img_B}

	def __len__(self):
		return len(self.datalist)

# data_root_dir = 'RIRE-ct-t1'
# mydata = MRI_T1_CT_Dataset(data_root_dir)
# # print(mydata[24]['A'].shape, mydata[24]['B'].shape)
# print(mydata[24])

# dataloader = DataLoader(
# 	MRI_T1_CT_Dataset("../../Processed_Data/%s" % data_root_dir),
# 	batch_size=opt.batch_size,
# 	shuffle=True,
# 	num_workers=opt.n_cpu,
# )

