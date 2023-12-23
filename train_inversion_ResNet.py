from __future__ import print_function
import argparse
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import BinaryDataset, extract_dataset
from model import Classifier, Inversion, LRmodule, LayerActivations, FaceResNet, ResBlock
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.datasets as dsets
import logging
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from Evasion import PGD, PGD_targeted, SimBA
from tools import compute_acccuracy, compute_success, imshow


import math
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=100, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=10)
parser.add_argument('--truncation', type=int, default=10)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=0, metavar='')
parser.add_argument('--path_out', type=str, default='vector-based/')
# parser.add_argument('--logfile', type=str, default='loss.log')
parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--inter_n', type=int, default=4)
parser.add_argument('--pre_accuracy', type=float, default=0.9)
parser.add_argument('--penalty', type=str, default='no')
parser.add_argument('--norm', type=str, default='all')
parser.add_argument('--eps_adv', type=float, default=0.4)
parser.add_argument('--classifier_path', type=str, default='Model_100/wo_adv_w_pub_wopre/')
parser.add_argument('--lambda_pen', type=float, default=0.001)
parser.add_argument('--adv_param', type=str, default='(500, 0.1)')




def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch, penalty, norm_l, eps, path_out, classifier_sub, lambda_adv, classifier_adv, adv_param):
    classifier.eval()
    classifier_sub.eval()
    inversion.train()
    # com_suc = 1
    plot = True

    success_rate = {'inf': -1, 1: -1, 2: -1}
    norm_list = []
    if  norm_l == 'all':
        norm_list = ['inf', 1, 2]
    elif norm_l == 'inf':
        norm_list = ['inf']
    elif norm_l == '1' or norm_l == '2':
        norm_list = [int(norm_l)]
    elif norm_l == 'noadv':
        norm_list = []
    elif norm_l == 'inf+2':
        norm_list = ['inf', 2]

    for batch_idx, (data, target) in enumerate(data_loader):
        # print("batch_idx:", batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)
            prob, pred_label = torch.max(prediction, dim=1)
        reconstruction = inversion(prediction)
        if penalty == 'no':
            loss = F.mse_loss(reconstruction, data)
            # print(" ================penalty is False===================")
        elif penalty == 'yes':
            # print(" penalty is true===================")
            loss = F.mse_loss(reconstruction, data) + lambda_adv*F.nll_loss(classifier_sub(reconstruction), pred_label.view_as(target))
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))            

        # ================Augment adversarial examples to training data of inversion model====================
        # print("norm:{}, \t norm_list:{}".format(norm_l, norm_list))
        for norm in norm_list:
            '''
            Only in the first epoch to generate adv data.
            '''

            if epoch == 1:
                '''
                PGD generates adv examples:
                '''
                # data_adv_target = PGD(classifier_sub, data, target, device, norm, eps=eps, alpha=2/255, iters=40)
                '''
                Simple black-box attacks of generating adv examples:
                '''
                with torch.no_grad():
                    attacker = SimBA(classifier, 'FaceScrub', 64)
                    max_iters = int(adv_param.split(',')[0][1:])
                    epsilon = float(adv_param.split(',')[1][:-1])
                    data_adv_target = attacker.simba_batch(
                        images_batch=data.cpu(), labels_batch=target.cpu(), max_iters=max_iters, epsilon=epsilon, linf_bound=0,
                        targeted=False, log_every=40)
                    print('succs_accurate:{}, l2_norms:{}, linf_norms:{}'.format(data_adv_target[4][:,-1].sum()/len(data), data_adv_target[6].mean(), data_adv_target[4][7].mean()))
                    # logger.info()
                
                os.makedirs('data_adv/', exist_ok=True)
                torch.save(data_adv_target, 'data_adv/adv_data_{}_{}.pt'.format(norm, batch_idx))
                

                truth_clean = data[0:32]
                truth_adv = data_adv_target[0][0:32].to(device)
                out = torch.cat((truth_clean, truth_adv))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = truth_adv[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth_clean[i * 8:i * 8 + 8]
                vutils.save_image(out, path_out + 'adv_vs_clean.png', nrow=8, normalize=True)

                # if batch_idx == 0:
                success_rate[norm] = compute_success(classifier, data_adv_target[0].to(device), data_adv_target[2].to(device), 0)
            else:
                data_adv_target = torch.load('data_adv/adv_data_{}_{}.pt'.format(norm, batch_idx))

            # print("data_adv_target.type:", type(data_adv_target))
            data_adv, target_adv, target_ori = data_adv_target[0].to(device), data_adv_target[1].to(device), data_adv_target[2].to(device)
            # print("data_adv.shape:", data_adv.shape)
            

            optimizer.zero_grad()
            with torch.no_grad():
                prediction_adv = classifier(data_adv, release=True)
            reconstruction_adv = inversion(prediction_adv)
            if penalty == 'no':
                loss = F.mse_loss(reconstruction_adv, data_adv)
            elif penalty == 'yes':
                loss = F.mse_loss(reconstruction_adv, data_adv) + lambda_adv*F.nll_loss(classifier_sub(reconstruction_adv), target_adv)
                # loss = F.mse_loss(reconstruction_adv, data_adv) + 0.001*F.nll_loss(classifier(reconstruction_adv), target)
            # loss = F.mse_loss(reconstruction_adv, data_adv) #- SSIM()(reconstruction, data_adv)
            loss.backward()
            optimizer.step()
        if epoch==1 and batch_idx >= 0:
            print('Success rate of AdvExample: \n[L_inf: {},\tL_1:{},\tL_2:{}'.format(success_rate['inf'], success_rate[1], success_rate[2]))
        if batch_idx % log_interval == 0:
            print('Train Epoch with AdvExample: {} [{}/{}]\tLoss: {:.6f}\n'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))
        if plot:
            truth = data[0:128]
            inverse = reconstruction[0:128]
            out = torch.cat((inverse, truth))
            for i in range(8):
                out[i * 32:i * 32 + 16] = inverse[i * 16:i * 16 + 16]
                out[i * 32 + 16:i * 32 + 32] = truth[i * 16:i * 16 + 16]
            vutils.save_image(out, path_out + 'recon_train_clean.png', nrow=16, normalize=True)
            
            if norm_list != []:
                truth = data_adv[0:128]
                inverse = reconstruction_adv[0:128]
                out = torch.cat((inverse, truth))
                for i in range(8):
                    out[i * 32:i * 32 + 16] = inverse[i * 16:i * 16 + 16]
                    out[i * 32 + 16:i * 32 + 32] = truth[i * 16:i * 16 + 16]
                vutils.save_image(out, path_out + 'recon_train_adv_normalize.png', nrow=16, normalize=True)
                vutils.save_image(out, path_out + 'recon_train_adv.png', nrow=16, normalize=False)

            plot = False                                               


def test(classifier, inversion, device, data_loader, epoch, msg, logger, path_out):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 64:i * 64 + 32] = inverse[i * 32:i * 32 + 32]
                    out[i * 64 + 32:i * 64 + 64] = truth[i * 32:i * 32 + 32]
                vutils.save_image(out, path_out + 'recon_round.png', nrow=32, normalize=True)
                plot = False

    mse_loss /= len(data_loader.dataset) * 64 * 64
    logger.info('\nTest inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, mse_loss))
    print('\nTest inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, mse_loss))
    return mse_loss


def main():
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=args.path_out + 'inv_loss.log',
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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda == False:
        logger.info('GPU is not used')
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])

    
    train_test_set = ImageFolder('../../FaceScrub/data100Person', transform=transform)
    # train_test_set = ImageFolder('../../FaceScrub/substitute100', transform=transform)
    # train_set_substitute, test_set_substitute = train_test_split(train_test_set_substitute, test_size=0.2, random_state=1)
    print("len of FaceScrub:", len(train_test_set))     

    train_set, test_set = train_test_split(train_test_set, test_size=0.2, random_state=1)
    print("len of Train_dataset:", len(train_set))
    print("len of Test_dataset:", len(test_set))
    logger.info("len of Train_dataset:".format(len(train_set)))
    logger.info("len of Test_dataset:".format(len(test_set)))

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
    logger.info("len of train_data_classifier:".format(len(train_data_classifier)))
    logger.info("len of test_data_classifier:".format(len(test_data_classifier)))

    # Use classifier data to train classifier
    train_loader = torch.utils.data.DataLoader(train_data_inversion, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data_inversion, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # Use full data to train classifier
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    classifier = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
    classifier_sub = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    # classifier_sub = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
    # Load classifier
    path = 'Model_100/ResNet_{}.pth'.format(args.pre_accuracy)
    path_sub = '{}classifier100_{}.pth'.format(args.classifier_path, args.pre_accuracy)

    checkpoint = torch.load(path)
    checkpoint_sub = torch.load(path_sub)

    classifier.load_state_dict(checkpoint['model'])
    # print("test success")
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    logger.info("=> loaded target classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))

    classifier_sub.load_state_dict(checkpoint_sub['model'])
    # print("test success")
    epoch_sub = checkpoint_sub['epoch']
    best_cl_acc_sub = checkpoint_sub['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path_sub, epoch_sub, best_cl_acc_sub))
    logger.info("=> loaded substitute classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path_sub, epoch_sub, best_cl_acc_sub))

    '''
    load a classifier to generate adversarial examples
    '''
    classifier_adv = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    # path_adv = 'Model_100/wo_adv_wo_pub/classifier100_{}.pth'.format(args.pre_accuracy)
    # checkpoint_adv = torch.load(path_adv)
    # classifier_adv.load_state_dict(checkpoint_adv['model'])
    # # print("test success")
    # epoch_adv = checkpoint_adv['epoch']
    # best_cl_acc_adv = checkpoint_adv['best_cl_acc']
    # print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch_adv, best_cl_acc_adv))
    # logger.info("=> loaded adv classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch_adv, best_cl_acc_adv))


    # Test the accuracy of the loaded classifier
    total = 0
    correct = 0

    classifier.eval()
    classifier_sub.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):

            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)

            prob, pred_label = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (pred_label == labels).sum().item()

    print("posi_num:{}, neg_num:{}".format(correct, total - correct))
    acc = correct * 1.0 / total
    print("target model accuracy: ", acc)

    if args.inter_n == 4:
        inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
        optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

        # Train inversion model using complete structure
        best_recon_loss = 999
        early_stop_label = 0
        for epoch in range(1, args.epochs + 1):
            train(classifier, inversion, args.log_interval, device, train_loader, optimizer, epoch, args.penalty, args.norm, args.eps_adv, args.path_out, classifier_sub, args.lambda_pen, classifier_adv, args.adv_param)
            recon_loss = test(classifier, inversion, device, test_loader, epoch, 'test1', logger, args.path_out)
            # test(classifier, inversion, device, test2_loader, epoch, 'test2')

            if recon_loss < best_recon_loss:
                best_recon_loss = recon_loss
                state = {
                    'epoch': epoch,
                    'model': inversion.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_recon_loss': best_recon_loss
                }
                torch.save(state, args.path_out + 'inversion.pth')
                shutil.copyfile(args.path_out + 'recon_round.png', args.path_out + 'best.png')
                # shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')

                early_stop_label = 0
            else:
                early_stop_label += 1
                if early_stop_label == args.early_stop:
                    logger.info('\nThe best test inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, best_recon_loss))
                    break

if __name__ == '__main__':
    main()