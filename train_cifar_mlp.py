import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import base_model as my_model
from utils import convert_grayScale


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

def train_mlp(opt,model,main_epoch):
    pre_model = my_model.gray_scale_net()
    training_logger_text = open('{0}/train_log_{1}.txt'.format(opt.experiment,main_epoch), 'w')
    val_logger_text = open('{0}/val_log_{1}.txt'.format(opt.experiment,main_epoch), 'w')
    mlp = my_model.mlp(outplane=opt.dimnoise)
    model.eval()
    pre_model.eval()
    pre_model.cuda()
    mlp.cuda()
    dataset = dsets.CIFAR10(root=opt.dataroot, train=True, download=False,
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                convert_grayScale(),
                                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]))
    assert dataset

    dataset_val = dsets.CIFAR10(root=opt.dataroot, train=False, download=False,
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                convert_grayScale(),
                                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]))
    assert dataset_val
    npoints = len(dataset)
    npoints_val = len(dataset_val)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=512, shuffle=False, num_workers=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), 0.005)

    model.cuda()

    total_numbatches = round(npoints/512)
    total_numbatches_val = round(npoints_val/512)
    statistics = np.ones((opt.nEpoch,4))
    for epoch in range(0, 20):
        model.train()
        mlp.train()
        data_iter = iter(dataloader)
        i = 0
        running_loss = 0
        running_Acc = 0
        while i < npoints:
            optimizer.zero_grad()
            data = data_iter.next()
            input = Variable(data[0].cuda())
            target = Variable(data[1].cuda())
            int_input = pre_model(input)
            noise_output = model(int_input)
            output = mlp(noise_output)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            prec1 = accuracy(output.data, target.data, topk=(1,))
            running_loss += loss.cpu().data.numpy()
            running_Acc += prec1[0].cpu().numpy()
            print('Train: [%d][%d/%d] Loss: [%f] Acc: [%f]' % (epoch, i, npoints, loss.cpu().data.numpy(), prec1[0].cpu().numpy()))

            i += 512
        epoch_loss = running_loss/total_numbatches
        epoch_acc = running_Acc/total_numbatches

        training_logger_text.write('{0} {1} \n'.format(epoch_loss,epoch_acc))
        print('Training summary: Epoch [%d] Loss: [%f] Acc: [%f]' % (epoch, epoch_loss, epoch_acc))
        statistics[epoch, 0] = epoch_loss
        statistics[epoch, 1] = epoch_acc

        model.eval()
        mlp.eval()
        data_iter = iter(dataloader_val)
        i = 0
        running_loss = 0
        running_Acc = 0
        while i < npoints_val:
            data = data_iter.next()
            input = Variable(data[0].cuda())
            target = Variable(data[1].cuda())
            int_input = pre_model(input)
            noise_output = model(int_input)
            output = mlp(noise_output)
            loss = criterion(output, target)
            prec1 = accuracy(output.data, target.data, topk=(1,))
            running_loss += loss.cpu().data.numpy()
            running_Acc += prec1[0].cpu().numpy()
            print('Val: [%d][%d/%d] Loss: [%f] Acc: [%f]' % (epoch, i, npoints, loss.cpu().data.numpy(), prec1[0].cpu().numpy()))

            i += 512
        epoch_loss = running_loss / total_numbatches_val
        epoch_acc = running_Acc / total_numbatches_val

        val_logger_text.write('{0} {1} \n'.format(epoch_loss, epoch_acc))
        print('Validation summary: Epoch [%d] Loss: [%f] Acc: [%f]' % (epoch, epoch_loss, epoch_acc))
        statistics[epoch, 2] = epoch_loss
        statistics[epoch, 3] = epoch_acc


