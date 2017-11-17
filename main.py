import torch.nn as nn
import argparse
import copy
import os


import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR as schedule_lr

import numpy as np
import train_cifar_mlp
from custom_sampler import NAT_sampler
from utils import rand_unit_sphere, calc_optimal_target_permutation,convert_grayScale
import base_model as my_model

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/mnt/data1/goh4hi/cifar10/')
parser.add_argument('--experiment', default='default_vgg/')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--num_ch', type=int, default=3)
parser.add_argument('--ngpu' , type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batchSize', type=int, default=256)
parser.add_argument('--nEpoch', type=int, default=300)
parser.add_argument('--dimnoise', type=int, default=100)
parser.add_argument('--resume', default='')
parser.add_argument('--resume_index', type=int)


opt = parser.parse_args()

#define model here
opt.experiment = '/mnt/data1/goh4hi/Nets/pytorch/{0}'.format(opt.experiment)
os.system('mkdir {0}'.format(opt.experiment))
# open logger text file
if opt.resume:
    logger_text = open('{0}/log.txt'.format(opt.experiment), 'a')
else:
    logger_text = open('{0}/log.txt'.format(opt.experiment), 'w')

#model = models.alexnet()
pre_model = my_model.gray_scale_net()
model = my_model.sanity_model(opt.dimnoise,in_planes=2)
for parameter in model.parameters():
    print(len(parameter))

npoints = 50000
# setup the dataloader and data sampler
index_list = torch.randperm(npoints)
my_sampler = NAT_sampler(index_list)
dataset = dsets.CIFAR10(root=opt.dataroot, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            convert_grayScale(),
                            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=my_sampler, num_workers=2)

criterion = nn.MSELoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.99)
def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

unit_sphere_noises_numpy = rand_unit_sphere(npoints, opt.dimnoise)
unit_sphere_noises = torch.from_numpy(unit_sphere_noises_numpy).float()
unit_sphere_noises = unit_sphere_noises[index_list]
start_from = 0
if opt.resume:
    print('Loading from checkpoint')
    start_from = opt.resume_index+1
    haal = torch.load('{0}/checkpoint_epoch_{1}.t7'.format(opt.experiment, opt.resume_index))
    model.load_state_dict(haal['model'])
    index_list = haal['index_list']
    unit_sphere_noises = haal['noise']
#=========================================
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# convert everything to cuda
pre_model.cuda()
model.cuda()
input = input.cuda()

scheduler = schedule_lr(optimizer, step_size=75, gamma=0.75)

total_numbatches = round(npoints/opt.batchSize)
loss_statistics = np.ones(opt.nEpoch)
for epoch in range(start_from,opt.nEpoch):
    scheduler.step()
    data_iter = iter(dataloader)
    i = 0
    running_loss = 0
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
        int_input = pre_model(input)
        output = model(int_input)

        # if statement is to enforce hungarian reassignment every x epoch
        if (epoch+1)%3 == 0:
        # calculate optimal target assignment within batch
            noise_in_this_batch = torch.from_numpy(calc_optimal_target_permutation(output.cpu().data.numpy(), noise_in_this_batch.numpy()))
        #update the global noise matrix Y
            for j in range(a):
                unit_sphere_noises[batch_indices[j], :] = noise_in_this_batch[j, :]

        targets = Variable(noise_in_this_batch.cuda())
        loss_y = criterion(output, targets)
        loss_y.backward()
        optimizer.step()

        i += a
        print('[%d][%d/%d] Loss: [%f]' % (epoch, i, npoints, loss_y.cpu().data.numpy()))
        running_loss += loss_y.cpu().data.numpy()
    index_list = torch.randperm(npoints)
    my_sampler.update(index_list)
    if (epoch + 1) % 3 == 0:
        print('Saving State')
        state = {
            'model': model.state_dict(),
            'noise': unit_sphere_noises,
            'index_list': index_list,
        }
        optim_params = {
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, '{0}/checkpoint_epoch_{1}.t7'.format(opt.experiment, epoch))
        torch.save(optim_params, '{0}/optim_params.t7'.format(opt.experiment))
    running_loss = running_loss/total_numbatches
    logger_text = open('{0}/log.txt'.format(opt.experiment), 'a')
    logger_text.write("%f\n" % running_loss)
    logger_text.close()
    print('Training summary: Epoch [%d] Loss: [%f]' % (epoch,  running_loss))

    if (epoch+1) %9 == 0:
        print('starting MLP training')
        train_cifar_mlp.train_mlp(copy.deepcopy(opt), copy.deepcopy(model), epoch)

'''
        a = data[0].numpy()[0].transpose(1,2,0)
        plt.figure()
        plt.imshow(data[0].numpy()[0][0])
        plt.pause(0.5)
        plt.imshow(a)
        plt.show()
        print(a.shape)
'''
