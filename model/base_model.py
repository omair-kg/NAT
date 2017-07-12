import torch
import torch.nn as nn
import torch.nn.parallel


class sanity_model(nn.Module):
    def __init__(self, outplane,in_planes=3):
        super(sanity_model, self).__init__()
        ndf = 64
        gain=nn.init.calculate_gain('relu')
        main = nn.Sequential()
        main.add_module('initia.conv.0', nn.Conv2d(in_planes, ndf, 3, 1, 1))
        #main.add_module('initial.bn.1', nn.BatchNorm2d(64, 1e-3))
        main.add_module('initial.relu.0', nn.ReLU())
        repeats = 4
        in_planes = ndf
        for idx in range(repeats):
            channel_num = ndf * (idx+1)
            for i in range(2):
                main.add_module('conv.{0}.{1}'.format(idx+1,i), nn.Conv2d(in_planes, channel_num, 3, 1, 1))
                in_planes = channel_num
                main.add_module('relu.{0}.{1}'.format(idx+1,i), nn.ReLU())
            if idx < repeats-1:
                main.add_module('conv.{0}.2'.format(idx+1), nn.Conv2d(in_planes, channel_num, 3, 2, 1))
                main.add_module('relu.{0}.2'.format(idx + 1), nn.ReLU())
        # weight init for main
        for layer in main.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal(layer.weight)
        #
        flatten_feats = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
        )
        # weight init for flatten_feats
        for layer in flatten_feats.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)
        #
        classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(2048, outplane),
        )
        # weight init for classifier
        for layer in classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)
        #
        self.main = main
        self.flatten_feats = flatten_feats
        self.classifier = classifier

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        x = self.flatten_feats(x)
        output = self.classifier(x)
        return output