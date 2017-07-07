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
import argparse
import model.base_model_eval as my_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/goh4hi/cifar10/')
parser.add_argument('--experiment', default='/home/goh4hi/noise_as_targets_fh9/')
parser.add_argument('--modelName', default='checkpoint_epoch_0.t7')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--num_ch', type=int, default=3)
parser.add_argument('--ngpu' , type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batchSize', type=float, default=4)
parser.add_argument('--nEpoch', type=float, default=100)

opt = parser.parse_args()

state = torch.load('{0}/{1}'.format(opt.experiment, opt.modelName))
model = my_model.sanity_model()
model.load_state_dict(state['model'])
model = model.cpu()
model.eval()

dataset = dsets.CIFAR10(root=opt.dataroot, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
dataset_vis = dsets.CIFAR10(root=opt.dataroot, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                        ]))
assert dataset
a_sampler = torch.utils.data.sampler.SequentialSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=a_sampler)

npoints = len(dataset)

feature_space = np.zeros((opt.batchSize, 2048))
data_iter = iter(dataloader)
i = 0
model = model.cuda()
while i < 1:
    data = data_iter.next()
    a = data[0].numpy().shape[0]
    input = Variable(data[0].cuda())
    output = model(input)
    print(output)
    feature_space[i:i+a,:] = output.cpu().data.numpy()
    i += opt.batchSize
print(feature_space[0,1])
