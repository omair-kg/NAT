import torch
import torch.nn as nn
import torch.nn.parallel


class sanity_model(nn.Module):
    def __init__(self, outplane,in_planes=3):
        super(sanity_model, self).__init__()
        ndf = 64
        gain=nn.init.calculate_gain('relu')
        main = nn.Sequential()
        num_blocks = 5
        do_rate = 0.3
        channel_num = ndf
        '''
        main.add_module('conv.init', nn.Conv2d(in_planes, channel_num, 3, 1, 1))
        main.add_module('relu.init', nn.ReLU(True))

        main.add_module('conv.0.0', nn.Conv2d(channel_num, channel_num, 3, 1,1))
        main.add_module('relu.0.0', nn.ReLU(True))
        main.add_module('conv.0.1', nn.Conv2d(channel_num, channel_num, 3, 1, 1))
        main.add_module('relu.0.1', nn.ReLU(True))
        main.add_module('conv.0.2', nn.Conv2d(channel_num, channel_num, 3, 2, 1))
        main.add_module('relu.0.2', nn.ReLU(True))
        in_planes = channel_num
        channel_num = 128
        main.add_module('conv.1.0', nn.Conv2d(in_planes, channel_num, 3, 1, 1))
        main.add_module('relu.1.0', nn.ReLU(True))
        main.add_module('conv.1.1', nn.Conv2d(channel_num, channel_num, 3, 1, 1))
        main.add_module('relu.1.1', nn.ReLU(True))
        main.add_module('conv.1.2', nn.Conv2d(channel_num, channel_num, 3, 2, 1))
        main.add_module('relu.1.2', nn.ReLU(True))

        in_planes = channel_num
        channel_num = 192
        main.add_module('conv.2.0', nn.Conv2d(in_planes, channel_num, 3, 1, 1))
        main.add_module('relu.2.0', nn.ReLU(True))
        main.add_module('conv.2.1', nn.Conv2d(channel_num, channel_num, 3, 1, 1))
        main.add_module('relu.2.1', nn.ReLU(True))
        main.add_module('conv.2.2', nn.Conv2d(channel_num, channel_num, 3, 2, 1))
        main.add_module('relu.2.2', nn.ReLU(True))

        in_planes = channel_num
        channel_num = 256
        main.add_module('conv.3.0', nn.Conv2d(in_planes, channel_num, 3, 1, 1))
        main.add_module('relu.3.0', nn.ReLU(True))
        main.add_module('conv.3.1', nn.Conv2d(channel_num, channel_num, 3, 1, 1))
        main.add_module('relu.3.1', nn.ReLU(True))
        '''
        main.add_module('conv.0', nn.Conv2d(in_planes, 64, 3, 1))
        main.add_module('relu.0', nn.ReLU(True))

        main.add_module('conv.1', nn.Conv2d(64, 128, 3, 1))
        main.add_module('relu.1', nn.ReLU(True))


        main.add_module('conv.2', nn.Conv2d(128, 128, 3, 2))
        main.add_module('relu.2', nn.ReLU(True))

        main.add_module('conv.3', nn.Conv2d(128, 256, 3, 2))
        main.add_module('relu.3', nn.ReLU(True))

        main.add_module('conv.4', nn.Conv2d(256, 256, 3, 1))
        main.add_module('relu.4', nn.ReLU(True))
        main.add_module('pool.4', nn.MaxPool2d(2, stride=2))

        classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024,256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256,outplane),
        )
        # init weights of network
        for layer in main.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal(layer.weight)

        for layer in classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)
        self.main = main
        self.classifier = classifier
        self.outplane = outplane


    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        output_dim = output.size(1)
        if output_dim == self.outplane:
            norm_output = torch.norm(output, 2, 1).unsqueeze(1)
            output = torch.div(output, norm_output)
        return output