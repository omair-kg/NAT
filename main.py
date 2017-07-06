import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np
import os
import argparse
from utils import rand_unit_sphere, calc_optimal_target_permutation
from custom_sampler import NAT_sampler
import model.base_model as my_model
import matplotlib.pyplot as plt

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/goh4hi/cifar10/')
parser.add_argument('--experiment', default='/home/goh4hi/noise_as_targets/')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--num_ch', type=int, default=3)
parser.add_argument('--ngpu' , type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batchSize', type=float, default=4)
parser.add_argument('--nEpoch', type=float, default=1)

opt = parser.parse_args()
#define model here
#os.system('mkdir {0}'.format(opt.experiment))
#model = models.alexnet()
model = my_model.sanity_model(3)


# setup the dataloader and data sampler
index_list = torch.randperm(50000)
my_sampler = NAT_sampler(index_list)
dataset = dsets.CIFAR10(root=opt.dataroot, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=my_sampler)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

criterion = nn.MSELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), opt.lr)

unit_sphere_noises_numpy = rand_unit_sphere(50000)
unit_sphere_noises = torch.from_numpy(unit_sphere_noises_numpy).float()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# convert everything to cuda
model.cuda()
input = input.cuda()


for epoch in range(opt.nEpoch):
    data_iter = iter(dataloader)
    i = 0
    while i < 1:
        data = data_iter.next()
        a = data[0].numpy().shape[0]
        input = Variable(data[0].cuda())
        # extract submatrix r from noise matrix Y
        noise_in_this_batch = unit_sphere_noises[i:i+a, :]
        # calculate output y_hat
        output = model(input)
        # calculate optimal target assignment within batch
        noise_in_this_batch = torch.from_numpy(calc_optimal_target_permutation(output.cpu().data.numpy(), noise_in_this_batch.numpy()))
        #update the global noise matrix Y
        unit_sphere_noises[i:i + a, :] = noise_in_this_batch
        targets = Variable(noise_in_this_batch.cuda())
        loss_y = criterion(output, targets)
        loss_y.backward()
        optimizer.step()
        '''
        a = data[0].numpy()[0].transpose(1,2,0)
        plt.figure()
        plt.imshow(data[0].numpy()[0][0])
        plt.pause(0.5)
        plt.imshow(a)
        plt.show()
        print(a.shape)
        '''
        i += opt.batchSize
    #index_list = torch.randperm(50000)
    #my_sampler.update(index_list)
