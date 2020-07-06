# MedicalGAN
 Working on Image to Image translation using conditional GAN.

# Todo list
Make a function for sampling single image in main
Sample the image from same index over and over to make a gif
Make function for test of the sample (over all data or half data? Don't know)
Implement mse, mae, lr_decay
Experiment with hyperparameters


# Some advice
In sample_images(), you can get discriminator output as well for the fake image. Do that! And display it on image itself along with the psnr, ssim.

make sample_single_image() function in pix2pix.py so that you can sample it after every epoch. You can look at a train image and a test image.
Each image should have written on it, its psnr, ssim, discriminator value.
Then make a gif with this!

# Checks
Currently check main.py again, tried something with sample_special_images()
Runtime warning about plt is sorted with doing plt.close() after save_trios(), check in next run.
Also check filenames should be starting with k0_...