# Conditional GAN for MRI to CT translation
 Using techniques like Image to Image translation for paired input input images, this work is successfully able to translate any MRI T1 input into a CT image in a supervised learning framework. 

# Methodology
Experiment with hyperparameters, like weightage given to different loss functions, reconstruction loss, perceptual loss from VGG16, and adversarial loss. Performing a search over each parameter keeping others fixed.

Some fixed hyperparameters are:
1. Learning rate: Here le = 2e-4 for 100 epochs, then linear learning rate decay till end ie. 200 epochs
2. Discriminator loss reduction rate: Set to 0.5. We have to half GAN loss before backprop so that discriminator does not train much faster than generator
3. Number of patches in discriminator: 16. 
4. Adam optimizer with beta_1 = 0.5 and beta_2 = 0.999
5. Number of neighbouring slices used as input: 1. If we use more than 1 input slice, then contextual information can be used by the generator.

# Some observations
1. Generator does not learn fast, maybe because of lots of loss applied to it
2. G and D Loss fluctuate a lot in most cases
3. Very small gain in accuracy is observed after 100 epochs, then it plateaus

# Future Work
Finding hyper-parameters is a hard problem using brute force. Some heuristic would help.

