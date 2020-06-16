import random
import os
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MRI_T1_CT_Dataset(Dataset):
	def __init__(self, root):
		data_dir = os.listdir(root)
		ct_fn = sorted(os.listdir(root + '/' + data_dir[0]))
		mr_fn = sorted(os.listdir(root + '/' + data_dir[1]))
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
					mr_in = mr_img_data[:,:,j]
					mr_in = torch.from_numpy(mr_in)
					ct_in = ct_img_data[:,:,j]
					ct_in = torch.from_numpy(ct_in)

					mr_max = torch.max(mr_in)
					ct_max = torch.max(ct_in)
					mr_min = torch.min(mr_in)
					ct_min = torch.min(ct_in)

					mr_final = torch.div( (mr_in - mr_min), (1.0 *(mr_max - mr_min)) )
					ct_final = torch.div( (ct_in - ct_min), (1.0 *(ct_max - ct_min)) )

					mr_final = mr_final.reshape(1, mr_final.shape[0], mr_final.shape[1])
					ct_final = ct_final.reshape(1, ct_final.shape[0], ct_final.shape[1])

					mr_final = mr_final.type(torch.FloatTensor)
					ct_final = ct_final.type(torch.FloatTensor)

					self.datalist.append([ mr_final, ct_final ])

	def __getitem__(self, index):
		[mr_img, ct_img] = self.datalist[index]
		return {"A": mr_img, "B": ct_img}

	def __len__(self):
		return len(self.datalist)
