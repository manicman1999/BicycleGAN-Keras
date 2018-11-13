# BicycleGAN-Keras
BicycleGAN Implemented In Keras

Slight Modifications:
  - Only 1 patch discriminator, with 16 8x8 patches
  - Uses CNNs for both E and D
  - Uses Pooling and Upsampling instead of strides
  - Modified learning rates


![alt text](https://raw.githubusercontent.com/manicman1999/BicycleGAN-Keras/Results/i48.png)
Results

(1 x Labels, 1 x Ground Truth, 6 x Generated Image)

Dependencies:
  - Numpy
  - Pillow
  - Matplotlib (Somewhat redundant)
  - Tensorflow
  - Keras

Be sure to load data into /data/ folder.

data/DomainA and data/DomainB are for image pairs.
data/DomainAs and data/DomainBs are for non-paired images.

Run pretrained model:
Run main.py, 100 sample sheets will be generated.

Train a new model:
Clear parameters in declaration of BicycleGAN near the end of main.py.
Comment or delete the line where an old model is loaded near the end of main.py - "model.load(6)".
Set train_model to true near the end of main.py.
Run main.py.
