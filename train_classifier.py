from __future__ import print_function
import argparse
from asyncio.log import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import torch.nn.functional as F
from model import Classifier, FaceResNet, ResBlock
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder
import logging
from sklearn.model_selection import train_test_split
from VGG import vgg11_bn

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=30, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=50, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--path_out', type=str, default='classifier/')
parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--structure', type=str, default='CNN4')


def train(classifier, log_interval, device, data_loader, optimizer, epoch, logger):
    classifier.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = classifier(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))
            logger.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))


def test(classifier, device, data_loader, logger):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    logger.info('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset)


def main():
    args = parser.parse_args()
    os.makedirs(args.path_out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=args.path_out + 'classifier_loss.log',
                        filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认果不写的话，就是追加模式
                        format=
                        # '%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s'
                        '%(asctime)s -[line:%(lineno)d]: %(message)s'
                        # 日志格式
                        )
    logger = logging.getLogger(__name__)

    logger.info("================================")
    logger.info(args)
    logger.info("================================")

    print("================================")
    print(args)
    print("================================")
    os.makedirs(args.path_out, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        print("GPU is not used")
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])
    train_test_set = ImageFolder('../../FaceScrub/data100Person', transform=transform)
    print("len of transformed_dataset:", len(train_test_set))
    train_set, test_set = train_test_split(train_test_set, test_size=0.2, random_state=42)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    print("len of Train_dataset:", len(train_set))
    print("len of Test_dataset:", len(test_set))
    logger.info("len of Train_dataset:{}".format(len(train_set)))
    logger.info("len of Test_dataset:{}".format(len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    '''
    Structure setting
    '''
    if args.structure == 'CNN4':
        # Classifier: 4-layer CNN
        classifier = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    elif args.structure == 'ResNet':
        # Classifier: ResNet
        classifier = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
    elif args.structure == 'VGG':
        # Classifier: VGG
        classifier = nn.DataParallel(vgg11_bn(pretrained=False, num_classes=args.nz, init_weights=False)).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    best_cl_acc = 0
    best_cl_epoch = 0
    early_stop_label = 0

    # Train classifier
    for epoch in range(1, args.epochs + 1):
        train(classifier, args.log_interval, device, train_loader, optimizer, epoch, logger)
        cl_acc = test(classifier, device, test_loader, logger)

        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_cl_acc': best_cl_acc,
            }
            # torch.save(state, args.path_out + 'classifier100_' + str(predefined_accuracy) + '.pth')
            torch.save(state, args.path_out + 'classifier.pth')
            
            early_stop_label = 0
        else:
            early_stop_label += 1
            if early_stop_label == args.early_stop:
                break

    print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))


if __name__ == '__main__':
    main()