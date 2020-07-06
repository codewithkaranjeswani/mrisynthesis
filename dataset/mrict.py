import random
import os
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MRI_T1_CT_Dataset(Dataset):
	def __init__(self, root, slices=1):
		self.slices = slices
		assert self.slices % 2 == 1, "self.slices must be odd!"
		es = int(self.slices/2) # floor operation
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
				ct_img_data = ct_img_data[0::2,0::2,mid-10-es:mid+10+es]
				ct_img_data = np.array(ct_img_data)
				mr_img = nib.load(root + '/' + data_dir[1] + '/' + mr_file)
				mr_img_data = mr_img.get_fdata()
				mid = (int)(mr_img_data.shape[2] / 2)
				mr_img_data = mr_img_data[:,:,mid-10-es:mid+10+es]
				mr_img_data = np.array(mr_img_data)
				assert mr_img_data.shape == ct_img_data.shape, "MRI and CT have different shapes"
				sh = mr_img_data.shape[0] # square input # sh = 256
				for j in range(0, 20):
					finmr = torch.Tensor(size=(self.slices, sh, sh))
					finct = torch.Tensor(size=(self.slices, sh, sh))

					mr_in = mr_img_data[:,:,j-es:j+es+1]
					mr_in = torch.from_numpy(mr_in)
					mr_in = mr_in.view(mr_in.size(0), mr_in.size(1), 2*es + 1)
					ct_in = ct_img_data[:,:,j-es:j+es+1]
					ct_in = torch.from_numpy(ct_in)
					ct_in = ct_in.view(ct_in.size(0), ct_in.size(1), 2*es + 1)

					for k in range(self.slices):
						mrh = mr_in[:,:,k]
						cth = ct_in[:,:,k]

						mr_max = torch.max(mrh)
						ct_max = torch.max(cth)
						mr_min = torch.min(mrh)
						ct_min = torch.min(cth)

						mr_final = torch.div( (mrh - mr_min), (1.0 *(mr_max - mr_min)) )
						ct_final = torch.div( (cth - ct_min), (1.0 *(ct_max - ct_min)) )

						mr_final = mr_final.reshape(1, mr_final.shape[0], mr_final.shape[1])
						ct_final = ct_final.reshape(1, ct_final.shape[0], ct_final.shape[1])

						mr_final = mr_final.type(torch.FloatTensor)
						ct_final = ct_final.type(torch.FloatTensor)

						finmr[k,:,:] = mr_final
						finct[k,:,:] = ct_final
					self.datalist.append([ finmr , finct ])

	def __getitem__(self, index, a2b=1):
		[mr_img, ct_img] = self.datalist[index]
		if a2b == 1:
			ct_img = ct_img[int(self.slices / 2)]
			ct_img = ct_img.view(1, ct_img.size(0), ct_img.size(1))
		elif a2b == 0:
			mr_img = mr_img[int(self.slices / 2)]
			mr_img = mr_img.view(1, mr_img.size(0), mr_img.size(1))
		else:
			print("Error, direction not 0 or 1!")
			exit(1)
		return {"A": mr_img, "B": ct_img}

	def __len__(self):
		return len(self.datalist)


# dataset = MRI_T1_CT_Dataset("../../../Processed_Data/%s" % "RIRE-ct-t1")

# print(len(dataset))
# print(dataset[0]['A'].size())
# print(dataset[0]['B'].size())

# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2)
# for ind in range(len(dataset)):
# 	ax1.imshow(dataset[ind]['A'][1].T, cmap="gray", origin="lower")
# 	ax2.imshow(dataset[ind]['B'].T, cmap="gray", origin="lower")
# 	ax1.set_title("MRI")
# 	ax2.set_title("CT")
# 	# plt.tight_layout()
# 	# plt.show()
# 	plt.savefig('../input/some_{0}_{1}.png'.format(int((ind) / 18) + 1, (ind) % 18 + 1), dpi=300)

