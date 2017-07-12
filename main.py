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

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/goh4hi/cifar10/')
parser.add_argument('--experiment', default='default/')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--num_ch', type=int, default=3)
parser.add_argument('--ngpu' , type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batchSize', type=int, default=256)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--dimnoise', type=int, default=100)


opt = parser.parse_args()

#define model here
opt.experiment = '/home/goh4hi/noise_as_targets/{0}'.format(opt.experiment)
os.system('mkdir {0}'.format(opt.experiment))
#model = models.alexnet()
model = my_model.sanity_model(opt.dimnoise)

npoints = 50000
# setup the dataloader and data sampler
index_list = torch.randperm(npoints)
my_sampler = NAT_sampler(index_list)
dataset = dsets.CIFAR10(root=opt.dataroot, train=True, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=my_sampler, num_workers=2)

criterion = nn.MSELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), opt.lr)


unit_sphere_noises_numpy = rand_unit_sphere(npoints, opt.dimnoise)
unit_sphere_noises = torch.from_numpy(unit_sphere_noises_numpy).float()
unit_sphere_noises = unit_sphere_noises[index_list]

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# convert everything to cuda
model.cuda()
input = input.cuda()

running_loss = 0
total_numbatches = round(npoints/opt.batchSize)
loss_statistics = np.ones(opt.nEpoch)
for epoch in range(0,opt.nEpoch):
    data_iter = iter(dataloader)
    i = 0
    while i < npoints:
        optimizer.zero_grad()
        data = data_iter.next()
        a = data[0].numpy().shape[0]
        input = Variable(data[0].cuda())
        # extract submatrix r from noise matrix Y
        batch_indices = index_list[i:i+a]
        noise_in_this_batch = torch.zeros((a, opt.dimnoise))
        # this loop will fetch the noise vector corresponding to each batch image. Egal in the first epoch
        for j in range(a):
            noise_in_this_batch[j, :] = unit_sphere_noises[batch_indices[j], :]
        # calculate output y_hat
        output = model(input)
        # if statement is to enforce hungarian reassignment every x epoch
        #if (epoch)%3 == 0:
        # calculate optimal target assignment within batch
        noise_in_this_batch = torch.from_numpy(calc_optimal_target_permutation(output.cpu().data.numpy(), noise_in_this_batch.numpy()))
        #update the global noise matrix Y
        for j in range(a):
            unit_sphere_noises[batch_indices[j], :] = noise_in_this_batch[j, :]

        targets = Variable(noise_in_this_batch.cuda())
        loss_y = criterion(output, targets)
        loss_y.backward()
        optimizer.step()

        i += opt.batchSize
        print('[%d][%d/%d] Loss: [%f]' % (epoch, i, npoints, loss_y.cpu().data.numpy()))
        running_loss += loss_y.cpu().data.numpy()
    if epoch % 3 == 0:
        index_list = torch.randperm(npoints)
        unit_sphere_noises = unit_sphere_noises[index_list]
        my_sampler.update(index_list)
        print('Saving State')
    state = {
           'model': model.state_dict(),
           'optimizer': optimizer.state_dict(),
            }
    torch.save(state, '{0}/checkpoint_epoch_{1}.t7'.format(opt.experiment, epoch))
    running_loss = running_loss/total_numbatches
    print('Training summary: Epoch [%d] Loss: [%f]' % (epoch,  running_loss))
    loss_statistics[epoch] = running_loss
    torch.save(loss_statistics, '{0}/loss_statistics.t7'.format(opt.experiment))

'''
        a = data[0].numpy()[0].transpose(1,2,0)
        plt.figure()
        plt.imshow(data[0].numpy()[0][0])
        plt.pause(0.5)
        plt.imshow(a)
        plt.show()
        print(a.shape)
'''
