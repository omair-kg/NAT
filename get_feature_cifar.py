import argparse
import os

import h5py
import numpy as np
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import base_model as my_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/goh4hi/cifar10/')
parser.add_argument('--experiment', default='default/')
parser.add_argument('--modelName', type=int)
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--num_ch', type=int, default=3)
parser.add_argument('--ngpu' , type=int, default=1)
parser.add_argument('--batchSize', type=float, default=256)
parser.add_argument('--nEpoch', type=float, default=100)
parser.add_argument('--dimnoise', type=int, default=100)

opt = parser.parse_args()
model_num = opt.modelName
opt.modelName = 'checkpoint_epoch_{0}.t7'.format(opt.modelName)
opt.experiment = '/home/goh4hi/noise_as_targets/{0}'.format(opt.experiment)
state = torch.load('{0}/{1}'.format(opt.experiment, opt.modelName))
model = my_model.sanity_model(opt.dimnoise)
model.load_state_dict(state['model'])
model = model.cpu()
model.eval()
samples = [10,20,30,60,70,90]#[1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]#
num_NN = 5
dataset = dsets.CIFAR10(root=opt.dataroot, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
dataset_vis = dsets.CIFAR10(root=opt.dataroot, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                        ]))
assert dataset
a_sampler = torch.utils.data.sampler.SequentialSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=a_sampler, num_workers=2)

npoints = len(dataset)

feature_space = np.zeros((npoints, 100))
data_iter = iter(dataloader)
i = 0
model = model.cuda()
while i < npoints:
    data = data_iter.next()
    a = data[0].numpy().shape[0]
    input = Variable(data[0].cuda())
    output = model(input)
    feature_space[i:i+a, :] = output.cpu().data.numpy()
    i += a
''' 
plt.figure()
plt.imshow(feature_space[1:1000,:])
plt.show()
'''
index_matrix = np.zeros((6, num_NN))
for counter, sample in enumerate(samples):
    feat_without = np.concatenate([feature_space[:sample, :], feature_space[sample+1:, :]], axis=0)
    for i in range(num_NN):
        nearest_index = np.sum(np.square(feat_without-feature_space[sample]), axis=1).argmin()
        index_matrix[counter, i] = nearest_index
        feat_without = np.concatenate([feat_without[:nearest_index, :], feat_without[nearest_index+1:, :]], axis=0)

#print(index_matrix)
print(index_matrix[:])
os.system('mkdir {0}/NN'.format(opt.experiment))
f = h5py.File('{0}/NN/indices_{1}.hdf5'.format(opt.experiment, model_num), 'w')
dset = f.create_dataset('/index', data=index_matrix[:])
f.close()
print('Finished')

