import matplotlib.pyplot as plt
import pandas as pd

diry =  '../output/results_07072020_2/UNet_10_0_1_output/'

dire = diry + 'csv/'
df = pd.read_csv(dire + 'k0_loss_pix2pix_UNet_200_train.csv', sep=',')
tf = pd.read_csv(dire + 'k0_loss_pix2pix_UNet_200_test.csv', sep=',')

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(df["epoch"], df["D loss"], label="D loss train")
ax1.plot(tf["epoch"], tf["D loss"], label="D loss test")
ax1.set_ylim([0,1])
ax1.legend()
ax2.plot(df["epoch"], df["G loss"], label="G loss train")
ax2.plot(tf["epoch"], tf["G loss"], label="G loss test")
ax2.set_ylim([0,15])
ax2.legend()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("D Loss")
ax1.set_title("D Loss decay for train and test")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("G Loss")
ax2.set_title("G Loss decay for train and test")
plt.tight_layout()
# plt.show()
plt.savefig(diry + 'loss_graph.png', dpi=300)
plt.close()
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(df["epoch"], df["psnr avg"], label="psnr avg train")
ax1.plot(tf["epoch"], tf["psnr avg"], label="psnr avg test")
ax1.set_ylim([10,35])
ax1.legend()
ax2.plot(df["epoch"], df["ssim avg"], label="ssim avg train")
ax2.plot(tf["epoch"], tf["ssim avg"], label="ssim avg test")
ax2.set_ylim([-0.25,1])
ax2.legend()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("PSNR (in dB)")
ax1.set_title("PSNR for train and test")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("SSIM")
ax2.set_title("SSIM for train and test")
plt.tight_layout()
# plt.show()
plt.savefig(diry + 'acc_graph.png', dpi=300)
plt.close()

