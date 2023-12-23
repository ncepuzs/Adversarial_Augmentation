from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False, celoss=False):

        x = x.view(-1, 1, 64, 64)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)
        if celoss:
            return x
        else:
            if release:
                return F.softmax(x, dim=1)
            else:
                return F.log_softmax(x, dim=1)


class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    # def transform_h(self, x):
    #     x_max, idx = torch.topk(x, 1)
    #     for i in range(len(x)):
    #         # x_max[i][0] = self.h[idx[i][0]]
    #         ave_val = (1 - self.h[idx[i][0]]) / (self.nz - 1)
    #         x_i_list = [ave_val]*self.nz
    #         x_i_list[idx[i]] = self.h[idx[i][0]]
    #         x_i_tensor = torch.tensor(x_i_list).cuda()
    #         x[i] = x_i_tensor

    #     return x

    # def one_hot(self, x):
    #     x_max, idx = torch.topk(x, 1)
    #     for i in range(len(x)):
    #         x_max[i][0] = 1
    #     x = torch.zeros(len(x), self.nz).cuda().scatter_(1, idx, x_max)

    #     return x

    def truncation_vector(self, x):
        top_k, indices = torch.topk(x, self.truncation)
        top_k = torch.clamp(torch.log(top_k), min=-1000) + self.c
        top_k_min = top_k.min(1, keepdim=True)[0]
        top_k = top_k + F.relu(-top_k_min)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, top_k)
        # x = torch.zeros(len(x), self.nz).scatter_(1, indices, top_k)

        return x

    def forward(self, x):
        # if self.truncation == -1:
        #     # our method
        #     x = self.transform_h(x)
        # elif self.truncation == 0:
        #     # one hot
        #     x = self.one_hot(x)
        # else:
        #     # vector-based or score-based
        #     x = self.truncation_vector(x)
        x = self.truncation_vector(x)
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x


# binary classifiers for one specific class
class LRmodule(nn.Module):
    def __init__(self, input_size):
        super(LRmodule, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        outdata = x.view(x.size(0), -1)
        outdata = self.fc(outdata)
        return torch.sigmoid(outdata)

# # ResNet18 defined
# class ResBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )
            
#     def forward(self, x):
#         out = self.left(x)
#         out = out + self.shortcut(x)
#         out = F.relu(out)
        
#         return out

# class FaceResNet(nn.Module):
#     def __init__(self, ResBlock, num_classes=10):
#         super(FaceResNet, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
#         self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)  
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))      
#         self.fc = nn.Linear(512, num_classes)
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         # print(*layers)
#         return nn.Sequential(*layers)
#     def forward(self, x, release=False, celoss=False):
#         x = x.view(-1, 1, 64, 64)
#         # if input size : 224x224
#         out = self.conv1(x) # output size: 112x112
#         out = self.maxpool(out) # output size: 56x56
#         out = self.layer1(out) # output size: 56x56
#         out = self.layer2(out) # output size: 28x28
#         out = self.layer3(out) # output size: 14x14
#         out = self.layer4(out) # output size: 7x7
#         out = self.avgpool(out) # output size: 1x1
#         out = out.view(out.size(0), -1)
#         out = self.fc(out) # output size: 1x1

#         if celoss:
#             return out
#         else:
#             if release:
#                 return F.softmax(out, dim=1)
#             else:
#                 return F.log_softmax(out, dim=1)



'''
Detail is referred to https://aisaka.cloud/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/pytorch%E8%8E%B7%E5%8F%96%E4%B8%AD%E9%97%B4%E5%B1%82%E5%8F%82%E6%95%B0%E3%80%81%E8%BE%93%E5%87%BA%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/
Class for hook

How to use:
# Define an instance
conv_out = LayerActivations(model.c1,0)
# Run the model
output = model(img)
# Unregistered the function
conv_out.remove()
# Get the intermediate representations from features variable
activations = conv_out.features
'''
class LayerActivations():
	# # Intermediate representation
	# features = None
	# Initialization. In the feedforward process, register_forward_hook is called.
	# Register_forward_hook will return self.hook
	def __init__(self,model,layer_num):
		self.hook = model[layer_num].register_forward_hook(self.hook_fn)
	# hook function: register_forward_hook requires 3 parameters in the hook_fn
	# module:the target model layer, input: input of the target layer, output: output of the target layer.
	def hook_fn(self,module,input,output):
		# Store the intermediate representations to features.
		self.features = output
	# Remove the self.hook
	def remove(self):
		self.hook.remove()



