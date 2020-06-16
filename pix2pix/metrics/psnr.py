import torch

def psnr(x, y, data_range=1.0):
	if (x.size() != y.size()):
		print("sizes of inputs are different in psnr function.")
	N = x.size()[0]
	mseloss = torch.nn.MSELoss(reduction='none')
	mse = mseloss(x, y)
	mse = torch.sum(mse, dim=(2, 3)) / (256*256)
	return 10 * torch.log10( (1.0*1.0) / mse ).view(N)
