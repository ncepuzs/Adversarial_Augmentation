import argparse
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
from VGG import vgg11, vgg11_bn, vgg16_bn


# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN

# signal(SIGPIPE, SIG_IGN)

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
parser.add_argument('--nz', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--path_out', type=str, default='classifier/')
parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--pre_accuracy', type=float, default=0.99)


def train(classifier, log_interval, device, data_loader, optimizer, epoch):
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


def test(classifier, device, data_loader):
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
    return correct / len(data_loader.dataset)


def main():
    args = parser.parse_args()
    os.makedirs(args.path_out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=args.path_out + 'evaluation_loss.log',
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

    # Dataset
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])

    train_test_set = ImageFolder('../../FaceScrub/data100Person', transform=transform)
    print("len of FaceScrub:", len(train_test_set))     

    train_set, test_set = train_test_split(train_test_set, test_size=0.2, random_state=1)
    print("len of Train_dataset:", len(train_set))
    print("len of Test_dataset:", len(test_set))
    logger.info("len of Train_dataset:{}".format(len(train_set)))
    logger.info("len of Test_dataset:{}".format(len(test_set)))

    # Split the train_dataset and test_dataset to classifier data and inversion data.
    # The rando_seed should be 1 to gurantee the same spliting results.
    len_train_classifier = len(train_set)//2
    len_train_inversion = len(train_set) - len(train_set)//2
    len_test_classifier = len(test_set)//2
    len_test_inversion = len(test_set) - len(test_set)//2
    train_data_classifier, train_data_inversion = torch.utils.data.random_split(train_set, [len_train_classifier,len_train_inversion], generator=torch.Generator().manual_seed(1))
    test_data_classifier, test_data_inversion = torch.utils.data.random_split(test_set, [len_test_classifier,len_test_inversion], generator=torch.Generator().manual_seed(1))
    print("len of train_data_classifier:", len(train_data_classifier))
    print("len of test_data_classifier:", len(test_data_classifier))
    logger.info("len of train_data_classifier:{}".format(len(train_data_classifier)))
    logger.info("len of test_data_classifier:{}".format(len(test_data_classifier)))

    # Use classifier data to train classifier
    train_loader = torch.utils.data.DataLoader(train_data_classifier, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data_classifier, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # Use full data to train classifier
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Classifier: encoder
    # classifier = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    # VGG11
    classifier = vgg11_bn(pretrained=False, num_classes=100, init_weights=False).to(device)
    # Classifier: ResNet
    # classifier = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    best_cl_acc = 0
    best_cl_epoch = 0
    early_stop_label = 0
    predefined_accuracy = args.pre_accuracy

    # Train classifier
    for epoch in range(1, args.epochs + 1):
        train(classifier, args.log_interval, device, train_loader, optimizer, epoch)
        cl_acc = test(classifier, device, test_loader)

        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_cl_acc': best_cl_acc,
            }
            torch.save(state, args.path_out + 'classifier100_' + str(predefined_accuracy) + '_16.pth')
            # if best_cl_acc > predefined_accuracy:
            #     break
            
            early_stop_label = 0
        else:
            early_stop_label += 1
            if early_stop_label == args.early_stop:
                break

    print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))


if __name__ == '__main__':
    main()