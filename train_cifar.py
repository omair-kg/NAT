import argparse
import itertools
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import base_model as my_model

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/goh4hi/cifar10/')
parser.add_argument('--experiment', default='cifar_baseline/')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--num_ch', type=int, default=3)
parser.add_argument('--ngpu' , type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batchSize', type=int, default=256)
parser.add_argument('--nEpoch', type=int, default=10)

opt = parser.parse_args()
opt.experiment = '/home/goh4hi/noise_as_targets/{0}'.format(opt.experiment)
os.system('mkdir {0}'.format(opt.experiment))
# open logger text file
training_logger_text = open('{0}/train_log.txt'.format(opt.experiment), 'w')
val_logger_text = open('{0}/val_log.txt'.format(opt.experiment), 'w')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

model = my_model.sanity_model(100)
mlp = my_model.mlp()

dataset = dsets.CIFAR10(root=opt.dataroot, train=True, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
assert dataset

dataset_val = dsets.CIFAR10(root=opt.dataroot, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
assert dataset_val
npoints = len(dataset)
npoints_val = len(dataset_val)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=2)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=2)


criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), opt.lr)
optimizer = optim.Adam(itertools.chain(model.parameters(), mlp.parameters()), lr=opt.lr)

model.cuda()
mlp.cuda()
total_numbatches = round(npoints/opt.batchSize)
total_numbatches_val = round(npoints_val/opt.batchSize)
statistics = np.ones((opt.nEpoch,4))
for epoch in range(0, 20):
    model.train()
    data_iter = iter(dataloader)
    i = 0
    running_loss = 0
    running_Acc = 0
    while i < npoints:
        optimizer.zero_grad()
        data = data_iter.next()
        input = Variable(data[0].cuda())
        target = Variable(data[1].cuda())
        noise_output = model(input)
        output = mlp(noise_output)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        prec1 = accuracy(output.data, target.data, topk=(1,))
        running_loss += loss.cpu().data.numpy()
        running_Acc += prec1[0].cpu().numpy()
        print('Train: [%d][%d/%d] Loss: [%f] Acc: [%f]' % (epoch, i, npoints, loss.cpu().data.numpy(), prec1[0].cpu().numpy()))

        i += opt.batchSize
    epoch_loss = running_loss/total_numbatches
    epoch_acc = running_Acc/total_numbatches

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, '{0}/checkpoint_epoch_{1}.t7'.format(opt.experiment, epoch))
    training_logger_text.write('{0} {1} \n'.format(epoch_loss,epoch_acc))
    print('Training summary: Epoch [%d] Loss: [%f] Acc: [%f]' % (epoch, epoch_loss, epoch_acc))
    #statistics[epoch, 0] = epoch_loss
    #statistics[epoch, 1] = epoch_acc

    model.eval()
    data_iter = iter(dataloader_val)
    i = 0
    running_loss = 0
    running_Acc = 0
    while i < npoints_val:
        data = data_iter.next()
        input = Variable(data[0].cuda())
        target = Variable(data[1].cuda())
        noise_output = model(input)
        output = mlp(noise_output)
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target.data, topk=(1,))
        running_loss += loss.cpu().data.numpy()
        running_Acc += prec1[0].cpu().numpy()
        print('Val: [%d][%d/%d] Loss: [%f] Acc: [%f]' % (epoch, i, npoints, loss.cpu().data.numpy(), prec1[0].cpu().numpy()))

        i += opt.batchSize
    epoch_loss = running_loss / total_numbatches_val
    epoch_acc = running_Acc / total_numbatches_val

    val_logger_text.write('{0} {1} \n'.format(epoch_loss, epoch_acc))
    print('Validation summary: Epoch [%d] Loss: [%f] Acc: [%f]' % (epoch, epoch_loss, epoch_acc))
    #statistics[epoch, 2] = epoch_loss
    #statistics[epoch, 3] = epoch_acc

    #f = h5py.File('{0}/stats.hdf5'.format(opt.experiment), 'w')
    #dset = f.create_dataset('/stats', data=statistics[:])
    #f.close()

