import torch
import torch.nn as nn
import torch.nn.parallel


class sanity_model(nn.Module):
    def __init__(self, in_planes):
        super(sanity_model, self).__init__()
        ndf = 64
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

        flatten_feats = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(True)
        )
        classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(2048, 100),
        )
        self.main = main
        self.flatten_feats = flatten_feats
        self.classifier = classifier

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        x = self.flatten_feats(x)
        output = self.classifier(x)
        return output