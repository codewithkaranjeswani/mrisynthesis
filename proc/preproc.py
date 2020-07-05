import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

def save_pairs(bulk, index):
	""" Function to display in a row, MRI and CT images """
	assert len(bulk) == 2, "Only send 20 MRI and 20 CT to save_pairs, please."
	for i in range(bulk[0].shape[2]):
		mr_slice = bulk[0][:,:,i]
		ct_slice = bulk[1][:,:,i]
		# print(i, mr_slice.shape, ct_slice.shape)
		fig, axes = plt.subplots(1, 2)
		axes[0].imshow(mr_slice.T, cmap="gray", origin="lower")
		axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
		# plt.show()
		plt.savefig(f'input/some_{index}_{i}.png')
		plt.close()

os.makedirs("input/", exist_ok=True)
data_root_dir = '../../Processed_Data/RIRE-ct-t1/'
data_dir = os.listdir(data_root_dir)
ct_fn = os.listdir(data_root_dir + data_dir[0])
mr_fn = os.listdir(data_root_dir + data_dir[1])

for i, (mr_file, ct_file) in enumerate(zip(mr_fn, ct_fn)):
	assert mr_file[:3] == ct_file[:3], "MRI and CT files not from the same patient"
	if (mr_file[:3] == ct_file[:3]):
		ct_img = nib.load(data_root_dir + data_dir[0] + '/' + ct_file)
		ct_img_data = ct_img.get_fdata()
		mid = (int)(ct_img_data.shape[2] / 2)
		ct_img_data = ct_img_data[0::2,0::2,mid-10:mid+10]

		mr_img = nib.load(data_root_dir + data_dir[1] + '/' + mr_file)
		mr_img_data = mr_img.get_fdata()
		mid = (int)(mr_img_data.shape[2] / 2)
		mr_img_data = mr_img_data[:,:,mid-10:mid+10]
		assert mr_img_data.shape == ct_img_data.shape, "MRI and CT have different shapes"
		# print(mr_file, mr_img_data.shape, ct_file, ct_img_data.shape)
		save_pairs([mr_img_data, ct_img_data], i)
