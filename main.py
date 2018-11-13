# -*- coding: utf-8 -*-
"""

@author: Matthew Mann

DomainA / DomainB = Images[] - Pairs
DomainAs / DomainBs = Amages[] - Non-Pairs

"""



import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from math import floor
import time

#Noise Level
nlev = 8

def zero_np():
    return np.random.uniform(0.0, 0.01, size = [1])

def one_np():
    return np.random.uniform(0.99, 1.0, size = [1])

def zero():
    return np.random.uniform(0.0, 0.01, size = [64])

def one():
    return np.random.uniform(0.99, 1.0, size = [64])

def noise(n):
    return np.random.normal(0, 1, size = [n, nlev])




#Import Images
print("Importing Images...")

#Import Pairs
ImagesA = []
ImagesB = []
n_images = 606

for n in range(1, n_images + 1):
    tempA = Image.open("data/DomainA/im ("+str(n)+").png")
    tempB = Image.open("data/DomainB/im ("+str(n)+").png")
    
    tempA1 = np.array(tempA.convert('RGB'), dtype='float32')
    tempB1 = np.array(tempB.convert('RGB'), dtype='float32')
    
    ImagesA.append(tempA1 / 255)
    ImagesB.append(tempB1 / 255)
    
    ImagesA.append(np.flip(ImagesA[-1], 1))
    ImagesB.append(np.flip(ImagesB[-1], 1))


#Import Non-paired
AmagesA = []
AmagesB = []
nimA = 606
nimB = 606

for n in range(1, nimA + 1):
    tempA = Image.open("data/DomainAs/im ("+str(n)+").png")
    
    tempA1 = np.array(tempA.convert('RGB'), dtype='float32')
    
    AmagesA.append(tempA1 / 255)
    
    AmagesA.append(np.flip(AmagesA[-1], 1))
    
for n in range(1, nimB + 1):
    tempB = Image.open("data/DomainBs/im ("+str(n)+").png")
    
    tempB1 = np.array(tempB.convert('RGB'), dtype='float32')
    
    AmagesB.append(tempB1 / 255)
    
    AmagesB.append(np.flip(AmagesB[-1], 1))


#Keras Imports
from keras.models import model_from_json, Model, Sequential
from keras.layers import Conv2D, LeakyReLU, AveragePooling2D, BatchNormalization, Reshape, Dense
from keras.layers import UpSampling2D, Activation, Dropout, concatenate, Input, Flatten, RepeatVector
from keras.optimizers import Adam
#import keras.backend as K


#Defining Layers For U-Net
def conv(input_tensor, filters, bn = True, drop = 0):
    
    co = Conv2D(filters = filters, kernel_size = 3, padding = 'same')(input_tensor)
    ac = LeakyReLU(0.2)(co)
    ap = AveragePooling2D()(ac)
    
    if bn:
        ap = BatchNormalization(momentum = 0.75)(ap)
        
    if drop > 0:
        ap = Dropout(drop)(ap)
    
    return ap

def deconv(input1, input2, filters, drop = 0):
    #Input 1 Should be half the size of Input 2
    up = UpSampling2D()(input1)
    co = Conv2D(filters = filters, kernel_size = 3, padding = 'same')(up)
    ac = Activation('relu')(co)
    
    if drop > 0:
        ac = Dropout(drop)(ac)
        
    ba = BatchNormalization(momentum = 0.75)(ac)
    con = concatenate([ba, input2])
    
    return con



#Define The Actual Model Class
class GAN(object):
    
    def __init__(self):
        
        #Always 256x256 Images
        
        #Models
        
        #Generator (Domain A -> Domain B)
        self.G1 = None
        
        #Generator (Domain B -> Domain A)
        self.G2 = None
        
        #Discriminator (Domain B)
        self.D = None
        
        #Encoder (Domain B -> Mapping)
        self.E = None
        
        #Old Models For Rollback After Training Others
        self.OD = None
        self.OG = None
        self.OE = None
        
        #Training Models
        self.DM = None #Discriminator Model (D)
        self.AM = None #Aversarial Model (G1 + D)
        self.VM = None #VAE Model (E + G1)
        self.VM2 = None #VAE Model 2 (E)
        self.ZM = None #Z Distribution Model (G1 + E)
        
        
        #Other Config
        self.LR = 0.00015 #Learning Rate
        self.steps = 1 #Training Steps Taken
    
    def generator1(self):
        
        #Defining G1 // U-Net
        if self.G1:
            return self.G1
        
        #Image Input
        inp_i = Input(shape = [256, 256, 3])
        
        #Noise Input
        inp_n = Input(shape = [nlev])
        #16
        rep1 = RepeatVector(65536)(inp_n)
        nv1 = Reshape(target_shape = [256, 256, nlev])(rep1)
        
        rep2 = RepeatVector(16384)(inp_n)
        nv2 = Reshape(target_shape = [128, 128, nlev])(rep2)
        
        rep3 = RepeatVector(4096)(inp_n)
        nv3 = Reshape(target_shape = [64, 64, nlev])(rep3)
        
        rep4 = RepeatVector(1024)(inp_n)
        nv4 = Reshape(target_shape = [32, 32, nlev])(rep4)
        #256x256x16
        
        inp = concatenate([inp_i, nv1])
        
        #256
        d0 = conv(inp, 16, False)
        d0 = concatenate([d0, nv2])
        #128
        d1 = conv(d0, 16)
        d1 = concatenate([d1, nv3])
        #64
        d2 = conv(d1, 32)
        d2 = concatenate([d2, nv4])
        #32
        d3 = conv(d2, 64)
        #16
        d4 = conv(d3, 128)
        #8
        d5 = conv(d4, 256)
        #4
        
        center = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(d5)
        ac = LeakyReLU(0.2)(center)
        
        #4
        u0 = deconv(ac, d4, 256)
        #8
        u1 = deconv(u0, d3, 128)
        #16
        u2 = deconv(u1, d2, 64)
        #32
        u3 = deconv(u2, d1, 32)
        #64
        u4 = deconv(u3, d0, 16)
        #128
        
        u5 = UpSampling2D()(u4)
        cc = concatenate([inp_i, u5])
        cl = Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu')(cc)
        #256
        out = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(cl)
        
        self.G1 = Model(inputs = [inp_i, inp_n], outputs = out)
        
        return self.G1
    
    def discriminator_np(self):
        
        #Defining D
        if self.D:
            return self.D
        
        #Image Input
        inp_a = Input(shape = [256, 256, 3])
        inp_b = Input(shape = [256, 256, 3])
        
        inp = concatenate([inp_a, inp_b])
        
        
        #256
        d0 = conv(inp, 8, False, drop = 0.25)
        #128
        d1 = conv(d0, 16, drop = 0.25)
        #64
        d2 = conv(d1, 32, drop = 0.25)
        #32
        d3 = conv(d2, 48, drop = 0.25)
        #16
        d4 = conv(d3, 64, drop = 0.25)
        #8
        d5 = conv(d4, 96, drop = 0.25)
        #4
        fl = Flatten()(d5)
        #4096 (Flat)
        d1 = LeakyReLU(0.2)(Dense(128)(fl))
        #512
        out = Dense(1, activation = 'linear')(d1)
        #Binary Output
        
        self.D = Model(inputs = [inp_a, inp_b], outputs = out)
        
        return self.D
    
    def discriminator(self):
        
        #Defining D
        if self.D:
            return self.D
        
        #Image Input
        inp_a = Input(shape = [256, 256, 3])
        inp_b = Input(shape = [256, 256, 3])
        
        inp = concatenate([inp_a, inp_b])
        
        #256
        d0 = conv(inp, 16, False, drop = 0.25)
        #128
        d1 = conv(d0, 32, drop = 0.25)
        #64
        d2 = conv(d1, 64, drop = 0.25)
        #32
        d3 = conv(d2, 128, drop = 0.25)
        #16
        d4 = conv(d3, 256, drop = 0.25)
        #16
        d5 = Conv2D(filters = 1, kernel_size = 3, padding = 'same', activation = 'linear')(d4)
        #Flatten, 16 8x8 patches
        fl = Flatten()(d5)
        #Binary Output
        
        self.D = Model(inputs = [inp_a, inp_b], outputs = fl)
        
        return self.D
    
    def encoder(self):
        
        #Defining E
        if self.E:
            return self.E
        
        
        #Image Input
        inp_b = Input(shape = [256, 256, 3])
        
        #256
        d0 = conv(inp_b, 8, False, drop = 0.2)
        #128
        d1 = conv(d0, 16, drop = 0.2)
        #64
        d2 = conv(d1, 32, drop = 0.2)
        #32
        d3 = conv(d2, 64, drop = 0.2)
        #16
        d4 = conv(d3, 128, drop = 0.2)
        #8
        d5 = conv(d4, 256, drop = 0.2)
        #4
        fl = Flatten()(d5)
        #4096 (Flat)
        d1 = Dense(256)(fl)
        #512
        out = Dense(nlev, activation = 'sigmoid')(d1)
        #Binary Output
        
        self.E = Model(inputs = inp_b, outputs = out)
        
        return self.E
    
    def DisModel(self):
        
        #Defining DM
        if self.DM == None:
            #Image A Input
            in1 = Input(shape = [256, 256, 3])
            #Image B Input
            in2 = Input(shape = [256, 256, 3])
            #D Part
            d = self.discriminator()([in1, in2])
            
            self.DM = Model(inputs = [in1, in2], outputs = d)
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.DM.compile(optimizer = Adam(lr = self.LR * 32 * (0.9 ** floor(self.steps / 10000))),
                        loss = 'mse')
        
        return self.DM
    
    def AdModel(self):
        
        #Defining AM
        if self.AM == None:
            #Image A Input
            in1 = Input(shape = [256, 256, 3])
            #Noise Input
            in2 = Input(shape = [nlev])
            #G1 Part
            g1 = self.generator1()([in1, in2])
            #D Part
            d = self.discriminator()([in1, g1])
            
            self.AM = Model(inputs = [in1, in2], outputs = d)
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.AM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))),
                        loss = 'mse')
        
        return self.AM
    
    def VAEModel(self):
        
        #Defining RM
        if self.VM == None:
            #Image A Input
            in1 = Input(shape = [256, 256, 3])
            #Image B Input
            in2 = Input(shape = [256, 256, 3])
            #G1 Part
            en = self.encoder()(in2)
            #G2 Part
            g1 = self.generator1()([in1, en])
            
            self.VM = Model(inputs = [in1, in2], outputs = g1)
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.VM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))),
                        loss = 'mean_squared_error')
        
        self.VM2 = Sequential()
        self.VM2.add(self.E)
        self.VM2.compile(optimizer = Adam(lr = self.LR * 0.1 * (0.9 ** floor(self.steps / 10000))),
                        loss = 'KLD')
        
        return self.VM
    
    def ZDModel(self):
        
        #Defining ZM
        if self.ZM == None:
            #Image Input
            in1 = Input(shape = [256, 256, 3])
            #Noise Input
            in2 = Input(shape = [nlev])
            #G1 Part
            g1 = self.generator1()([in1, in2])
            #Encoder Part
            en = self.encoder()(g1)
            
            self.ZM = Model(inputs = [in1, in2], outputs = en)
            
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.ZM.compile(optimizer = Adam(lr = self.LR * 0.5 * (0.9 ** floor(self.steps / 10000))),
                        loss = 'mean_squared_error')
        
        return self.ZM

    def sod(self):
        
        #Save Old Discriminator
        self.OD = self.D.get_weights()
    
    def lod(self):
        
        #Load Old Discriminator
        self.D.set_weights(self.OD)
        
    def soe(self):
        
        #Save Old Discriminator
        self.OE = self.E.get_weights()
    
    def loe(self):
        
        #Load Old Discriminator
        self.E.set_weights(self.OE)
        
    def sog(self):
        
        #Save Old Discriminator
        self.OG = self.G1.get_weights()
    
    def log(self):
        
        #Load Old Discriminator
        self.G1.set_weights(self.OG)
        

#Now Define The Actual Model
class BicycleGAN(object):
    
    def __init__(self, steps = -1):
        
        #Models
        #Main
        self.GAN = GAN()
        
        #Set Steps, If Relevant
        if steps >= 0:
            self.GAN.steps = steps
        
        #Generators
        self.G1 = self.GAN.generator1()
        
        self.GAN.encoder()
        #self.GAN.E.summary()
        #self.G1.summary()
        self.GAN.discriminator()
        #self.GAN.D.summary()
        
        #Training Models
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.VAEModel = self.GAN.VAEModel()
        self.ZDModel = self.GAN.ZDModel()
        self.lastblip = time.clock()
        
    def train(self, batch = 16):
        
        #Train and Get Losses
        (al, bl) = self.train_dis(batch)
        cl = self.train_ad(batch)
        (dl, nl) = self.train_vae(batch)
        el = self.train_zd(batch)
        
        #Every 20 Steps Display Info
        if self.GAN.steps % 20 == 0:
            ti = round((time.clock() - self.lastblip) * 100.0) / 100.0
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D Real: " + str(al))
            print("D Fake: " + str(bl))
            print("G1 D::: " + str(cl))
            print("VAE:::: " + str(dl))
            print("VAE Ns: " + str(nl))
            print("G1 E::: " + str(el))
            print("Time::: " + str(ti))
            self.lastblip = time.clock()
        
        #Save Every 500 steps
        if self.GAN.steps % 500 == 0:
            self.save(floor(self.GAN.steps / 10000))
            #self.evaluate()
        
        #Re-Compile (Update Learning Rate) Every 10k Steps
        if self.GAN.steps % 10000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.GAN.TM = None
            self.GAN.ZM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()
            self.VAEModel = self.GAN.VAEModel()
            self.ZDModel = self.GAN.ZDModel()
        
        self.GAN.steps = self.GAN.steps + 1
        
        return True
    
    def train_dis(self, batch): #Train Discriminator on Real, then Generated Images
        
        number = max(1, int(batch / 2))
        
        #Get Real Images
        train_data_a = []
        train_data_b = []
        label_data = []
        for i in range(number):
            im_no = random.randint(0, len(ImagesA) - 1)
            train_data_a.append(ImagesA[im_no])
            train_data_b.append(ImagesB[im_no])
            label_data.append(one())
        
        train_data_a = np.array(train_data_a)
        train_data_b = np.array(train_data_b)
            
        d_loss_real = self.DisModel.train_on_batch([train_data_a, train_data_b], np.array(label_data))
        
        #Get Fake Images
        train_data_a = []
        train_data_b = []
        label_data = []
        
        for i in range(number):
            im_no = random.randint(0, len(AmagesA) - 1)
            train_data_a.append(AmagesA[im_no])
            train_data_b.append(self.G1.predict([np.array([AmagesA[im_no]]), noise(1)])[0])
            label_data.append(zero())
            
        train_data_a = np.array(train_data_a)
        train_data_b = np.array(train_data_b)
            
        d_loss_fake = self.DisModel.train_on_batch([train_data_a, train_data_b], np.array(label_data))
        
        del train_data_a, train_data_b, label_data
        
        return (d_loss_real, d_loss_fake)
    
    def train_ad(self, batch): #Train Generator on Discriminator Adversarially
        
        #Save Old Discriminator
        self.GAN.sod()
        
        #Labels And Train Images
        label_data = []
        train_data = []
        for i in range(int(batch)):
            im_no = random.randint(0, len(ImagesA) - 1)
            train_data.append(AmagesA[im_no])
            label_data.append(one())
        
        g_loss = self.AdModel.train_on_batch([np.array(train_data), noise(batch)], np.array(label_data))
        
        #Load Old Discriminator
        self.GAN.lod()
        
        del train_data, label_data
        
        return g_loss
    
    def train_vae(self, batch): #Train VAE Model, including KL-Divergence
        
        #Labels And Train Images
        label_data = []
        train_data_1 = []
        train_data_2 = []
        for i in range(int(batch)):
            im_no = random.randint(0, len(ImagesA) - 1)
            train_data_1.append(ImagesA[im_no])
            train_data_2.append(ImagesB[im_no])
            label_data.append(ImagesB[im_no])
        
        
        g_loss = self.VAEModel.train_on_batch([np.array(train_data_1), np.array(train_data_2)], np.array(label_data))
        
        #Train KL Divergence
        g_loss_2 = self.GAN.VM2.train_on_batch(np.array(train_data_1), noise(batch))
        
        del train_data_1, train_data_2, label_data
        
        return (g_loss, g_loss_2)
    
    def train_zd(self, batch): #Train Z Distribution (cLR) Model
        
        #Labels And Train Images
        self.GAN.soe()
        
        label_data = []
        train_data = []
        for i in range(int(batch)):
            im_no = random.randint(0, len(AmagesA) - 1)
            train_data.append(AmagesA[im_no])
            
        label_data = noise(batch)
        
        g_loss = self.ZDModel.train_on_batch([np.array(train_data), label_data], label_data)
        
        self.GAN.loe()
        
        del train_data, label_data
        
        return g_loss
        
    def evaluate(self, show = True, test = False):
        
        if test:
            im_no = random.randint(0, len(AmagesA) - 1)
            im1 = AmagesA[im_no]
            im2 = AmagesA[im_no]
        else:
            im_no = random.randint(0, len(ImagesA) - 1)
            im1 = ImagesA[im_no]
            im2 = ImagesB[im_no]
        
        im3 = self.G1.predict([np.array([im1]), noise(1)])
        
        im4 = self.G1.predict([np.array([im1]), noise(1)])
        
        im5 = np.concatenate([im1, im2], axis = 1)
        im6 = np.concatenate([im3[0], im4[0]], axis = 1)
        
        im7 = np.concatenate([im5, im6], axis = 0)
        
        if show:
            plt.figure(1)
            plt.imshow(im5)
            plt.figure(2)
            plt.imshow(im6)
            plt.show()
            
        del im1, im2, im3, im4, im5, im6
        
        return im7
    
    def eval2(self, num):
        
        im = []
        
        #Blank Space
        brow = np.zeros(shape = [16, 2096, 3])
        bcol = np.zeros(shape = [512, 16, 3])
        
        #Get 12 Tries
        for _ in range(4):
            im.append(self.evaluate(False))
            im.append(bcol)
            
        for _ in range(8):
            im.append(self.evaluate(False, True))
            im.append(bcol)
        
        #Concatenate Rows
        row1 = np.concatenate(im[:7], axis = 1)
        row2 = np.concatenate(im[8:15], axis = 1)
        row3 = np.concatenate(im[16:23], axis = 1)
        
        image = np.concatenate([row1, brow, row2, brow, row3], axis = 0)
        
        x = Image.fromarray(np.uint8(image*255))
        
        x.save("Results/i"+str(num)+".png")
        
        del row1, row2, row3, image, x
        
    def eval3(self, num): #This is the evaluation function used
        
        row = []
        
        blank = np.zeros([256, 256, 3])
        
        #From left to right: Labels, GT, 6xGenerated Images
        
        #With Matching Ground Truth Image
        for _ in range(8):
            im_no = random.randint(0, len(ImagesA) - 1)
            imA = ImagesA[im_no]
            imB = ImagesB[im_no]
            out = self.G1.predict([np.array([imA, imA, imA, imA, imA, imA]), noise(6)])
            s = np.concatenate([imA, imB, out[0], out[1], out[2], out[3], out[4], out[5]], axis = 1)
            row.append(s)
            
        #No Matching Ground Truth Image
        for _ in range(0):
            im_no = random.randint(0, len(AmagesA) - 1)
            imA = ImagesA[im_no]
            imB = blank
            out = self.G1.predict([np.array([imA, imA, imA, imA, imA, imA]), noise(6)])
            row.append(np.concatenate([imA, imB, out[0], out[1], out[2], out[3], out[4], out[5]], axis = 1))
            
            
        
        image = np.concatenate(row[0:8], axis = 0)
        
        x = Image.fromarray(np.uint8(image*255))
        
        x.save("Results/i"+str(num)+".png")
        
        del row, image, x
            
    
    def save(self, num): #Save JSON and Weights into /Models/
        gen1_json = self.GAN.G1.to_json()
        dis_json = self.GAN.D.to_json()
        enc_json = self.GAN.E.to_json()

        with open("Models/gen1.json", "w") as json_file:
            json_file.write(gen1_json)
        
        with open("Models/dis.json", "w") as json_file:
            json_file.write(dis_json)
            
        with open("Models/enc.json", "w") as json_file:
            json_file.write(enc_json)

        self.GAN.G1.save_weights("Models/gen1_"+str(num)+".h5")
        self.GAN.D.save_weights("Models/dis"+str(num)+".h5")
        self.GAN.E.save_weights("Models/enc"+str(num)+".h5")

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Generator
        gen_file = open("Models/gen1.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()
        
        self.GAN.G1 = model_from_json(gen_json)
        self.GAN.G1.load_weights("Models/gen1_"+str(num)+".h5")
        
        #Discriminator
        dis_file = open("Models/dis.json", 'r')
        dis_json = dis_file.read()
        dis_file.close()
        
        self.GAN.D = model_from_json(dis_json)
        self.GAN.D.load_weights("Models/dis"+str(num)+".h5")
        
        #Encoder
        enc_file = open("Models/enc.json", 'r')
        enc_json = enc_file.read()
        enc_file.close()
        
        self.GAN.E = model_from_json(enc_json)
        self.GAN.E.load_weights("Models/enc"+str(num)+".h5")
        
        self.GAN.steps = steps1

        #Reinitialize
        self.G1 = self.GAN.generator1()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.VAEModel = self.GAN.VAEModel()
        self.ZDModel = self.GAN.ZDModel()

#Finally Onto The Main Function
model = BicycleGAN(62000)
model.load(6)

train_model = False

while(train_model):
    model.train(4)
    
    #Evaluate Every 1k Steps
    if model.GAN.steps % 1000 == 0:
        model.eval3(floor(model.GAN.steps / 1000))


#Evaluate 100x on loaded model
for i in range(100):
    model.eval3(i)

