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
from image_utils import SSIM
import pytorch_ssim


import math
import numpy as np

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

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
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--truncation', type=int, default=100)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=0, metavar='')
parser.add_argument('--path_out', type=str, default='vector-based/')
parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--penalty', type=str, default='no')
parser.add_argument('--norm', type=str, default='all')
parser.add_argument('--lambda_pen', type=float, default=0.001)
parser.add_argument('--adv_param', type=str, default='(500, 0.1)')
parser.add_argument('--k_top', type=int, default=2)
parser.add_argument('--train_or_not', type=str, default='train')
parser.add_argument('--victim', type=str, default='CNN')
parser.add_argument('--shadow', type=str, default='ResNet')
parser.add_argument('--loss_mode', type=str, default='MSE')







def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch, penalty, norm_l, path_out, classifier_sub, lambda_adv, adv_param, logger, k_top, train_or_not, loss_mode):
    classifier.eval()
    classifier_sub.eval()
    inversion.train()
    # com_suc = 1
    plot = True

    success_rate = {'inf': -1, 1: -1, 2: -1}
    norm_list = []
    success_rate_avg = 0
    l2_norm = 0
    linf_norm = 0
    if norm_l == '1' or norm_l == '2':
        norm_list = [int(norm_l)]
    elif norm_l == 'noadv':
        norm_list = []

    ssim_loss = pytorch_ssim.SSIM()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)
            prob, pred_label = torch.max(prediction, dim=1)
            _, pred_label_pseudo = torch.topk(prediction, k_top, dim=1)
        reconstruction = inversion(prediction)
        if penalty == 'no':
            if loss_mode == 'PSNR':
                mse = F.mse_loss(reconstruction, data)
                loss = -20 * torch.log10(255.0 / torch.sqrt(mse))
                # loss =  10 * torch.log10(1.0 / mse)
            elif loss_mode == 'SSIM':
                # loss = compare_ssim(reconstruction, data, data_range=255)
                # loss = -SSIM()(reconstruction, data)
                
                loss = -ssim_loss(reconstruction, data)
            else:
                loss = F.mse_loss(reconstruction, data)
            # print(" ================penalty is False===================")
        elif penalty == 'yes':
            if loss_mode == 'PSNR':
                mse = F.mse_loss(reconstruction, data)
                loss1 = -20 * torch.log10(255.0 / torch.sqrt(mse))
            elif loss_mode == 'SSIM':
                # loss1 = compare_ssim(reconstruction, data, data_range=255)
                # loss1 = -SSIM()(reconstruction, data)
                loss1 = -ssim_loss(reconstruction, data)
            else:
                loss1 = F.mse_loss(reconstruction, data)
            # print(" penalty is true===================")
            # loss = F.mse_loss(reconstruction, data) + lambda_adv*F.nll_loss(classifier_sub(reconstruction), pred_label.view_as(target))
            loss = loss1 + lambda_adv*F.cross_entropy(classifier_sub(reconstruction, celoss=True), classifier_sub(data, release=True))

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
                if train_or_not == 'train':
                    with torch.no_grad():
                        attacker = SimBA(classifier, 'FaceScrub', 64, pseudo=True, k_top=k_top)
                        max_iters = int(adv_param.split(',')[0][1:])
                        epsilon = float(adv_param.split(',')[1][:-1])
                        data_adv_target = attacker.simba_batch(
                            # images_batch=data.cpu(), labels_batch=target.cpu(), max_iters=max_iters, epsilon=epsilon, linf_bound=0,
                            images_batch=data.cpu(), labels_batch=pred_label_pseudo.cpu(), max_iters=max_iters, epsilon=epsilon, linf_bound=0,
                            targeted=False, log_every=40)
                        print('succs_accurate:{}, l2_norms:{}, linf_norms:{}'.format(data_adv_target[4][:,-1].sum()/len(data), data_adv_target[6].mean(), data_adv_target[4][7].mean()))
                        # logger.info('succs_accurate:{}, l2_norms:{}, linf_norms:{}'.format(data_adv_target[4][:,-1].sum()/len(data), data_adv_target[6].mean(), data_adv_target[4][7].mean()))
                        success_rate_avg += data_adv_target[4][:,-1].sum()/len(data)
                        l2_norm += data_adv_target[6].norm(2, 1).sum()/len(data)
                        linf_norm += data_adv_target[7].abs().max(1)[0].sum()/len(data)
                    # logger.info()
                
                    os.makedirs(path_out + 'data_adv/', exist_ok=True)
                    torch.save(data_adv_target, path_out + 'data_adv/adv_data_{}_{}.pt'.format(norm, batch_idx))
                elif train_or_not == 'load':
                    data_adv_target = torch.load(path_out + 'data_adv/adv_data_{}_{}.pt'.format(norm, batch_idx))
                

                truth_clean = data[0:32]
                truth_adv = data_adv_target[0][0:32].to(device)
                out = torch.cat((truth_clean, truth_adv))
                if truth_adv.shape[0]>=32:
                    for i in range(4):
                        out[i * 16:i * 16 + 8] = truth_adv[i * 8:i * 8 + 8]
                        out[i * 16 + 8:i * 16 + 16] = truth_clean[i * 8:i * 8 + 8]
                    vutils.save_image(out, path_out + 'adv_vs_clean_nor.png', nrow=8, normalize=True)
                    vutils.save_image(out, path_out + 'adv_vs_clean.png', nrow=8, normalize=False)

                # if batch_idx == 0:
                success_rate[norm] = compute_success(classifier, data_adv_target[0].to(device), data_adv_target[2].to(device), 0)
            else:
                data_adv_target = torch.load(path_out + 'data_adv/adv_data_{}_{}.pt'.format(norm, batch_idx))

            # print("data_adv_target.type:", type(data_adv_target))
            data_adv, target_adv, target_ori = data_adv_target[0].to(device), data_adv_target[1].to(device), data_adv_target[2].to(device)
            # print("data_adv.shape:", data_adv.shape)
            

            optimizer.zero_grad()
            with torch.no_grad():
                prediction_adv = classifier(data_adv, release=True)
            reconstruction_adv = inversion(prediction_adv)
            if penalty == 'no':
                if loss_mode == 'PSNR':
                    mse = F.mse_loss(reconstruction_adv, data_adv)
                    loss = -20 * torch.log10(255.0 / torch.sqrt(mse))
                elif loss_mode == 'SSIM':
                    # loss = -SSIM()(reconstruction, data)
                    loss = -ssim_loss(reconstruction_adv, data_adv)
                else:
                    loss = F.mse_loss(reconstruction_adv, data_adv)
            elif penalty == 'yes':
                if loss_mode == 'PSNR':
                    mse = F.mse_loss(reconstruction_adv, data_adv)
                    loss1 = -20 * torch.log10(255.0 / torch.sqrt(mse))
                elif loss_mode == 'SSIM':
                    # loss1 = -SSIM()(reconstruction, data)
                    loss1 = -ssim_loss(reconstruction_adv, data_adv)
                else:
                    loss1 = F.mse_loss(reconstruction_)
                    # loss = F.mse_loss(reconstruction_adv, data_adv) + lambda_adv*F.nll_loss(classifier_sub(reconstruction_adv), target_adv)
            loss = loss1 + lambda_adv*F.cross_entropy(classifier_sub(reconstruction_adv, celoss=True), classifier_sub(data_adv, release=True))
                    # loss = F.mse_loss(reconstruction_adv, data_adv) + 0.001*F.nll_loss(classifier(reconstruction_adv), target)
            # loss = F.mse_loss(reconstruction_adv, data_adv) #- SSIM()(reconstruction, data_adv)
            loss.backward()
            optimizer.step()
        if epoch==1 and batch_idx >= 0 and len(norm_list)>0:
            print('Success rate of AdvExample: \n[L_inf: {},\tL_1:{},\tL_2:{}'.format(success_rate['inf'], success_rate[1], success_rate[2]))
        if batch_idx % log_interval == 0:
            print('Train Epoch with AdvExample: {} [{}/{}]\tLoss: {:.6f}\n'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))
        if plot:
            truth = data[0:128]
            inverse = reconstruction[0:128]
            out = torch.cat((inverse, truth))
            if truth.shape[0]>=128:
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
    if epoch == 1:
        logger.info("The performance of adversarial examples:\n\
            adv_param:{}\n\
            success_rate:{},\n\
            l2_norm:{},\n\
            linf_norm:{}".format(adv_param, success_rate_avg/len(data_loader), \
            l2_norm/len(data_loader), linf_norm/len(data_loader)))                                               


def test(classifier, inversion, device, data_loader, epoch, msg, logger, path_out, data_loader_pri):
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
                vutils.save_image(out, path_out + 'recon_round_attack.png', nrow=32, normalize=True)
                plot = False

    mse_loss /= len(data_loader.dataset) * 64 * 64
    logger.info('\nTest inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, mse_loss))
    print('\nTest inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, mse_loss))

    mse_loss_pri = 0
    plot =True
    with torch.no_grad():
        for data, target in data_loader_pri:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss_pri += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 64:i * 64 + 32] = inverse[i * 32:i * 32 + 32]
                    out[i * 64 + 32:i * 64 + 64] = truth[i * 32:i * 32 + 32]
                vutils.save_image(out, path_out + 'recon_round_private.png', nrow=32, normalize=True)
                plot = False

    mse_loss_pri /= len(data_loader_pri.dataset) * 64 * 64
    logger.info('\nTest inversion model on {} epoch: Average MSE loss [Private]: {:.6f}\n'.format(epoch, mse_loss_pri))
    print('\nTest inversion model on {} epoch: Average MSE loss [Private]: {:.6f}\n'.format(epoch, mse_loss_pri))

    return mse_loss


def main():
    args = parser.parse_args()

    os.makedirs('Inversion_Models/' + args.path_out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename='Inversion_Models/'+args.path_out + 'inv_loss.log',
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

    
    pri_set = ImageFolder('../../FaceScrub/data100Person', transform=transform)
    train_test_set = ImageFolder('../../FaceScrub/substitute100', transform=transform)
    train_pri, test_pri = train_test_split(pri_set, test_size=0.2, random_state=42)
    print("len of FaceScrub:", len(train_test_set))     

    train_set, test_set = train_test_split(train_test_set, test_size=0.2, random_state=42)
    print("len of Train_dataset:", len(train_set))
    print("len of Test_dataset:", len(test_set))
    logger.info("len of Train_dataset:".format(len(train_set)))
    logger.info("len of Test_dataset:".format(len(test_set)))

    # Use classifier data to train classifier
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    pri_loader = torch.utils.data.DataLoader(test_pri, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    '''
    Structure setting
    '''
    if args.victim == 'CNN4':
        # Classifier: 4-layer CNN
        classifier = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
        path = 'Models/Classifier/CNN/classifier.pth'

        if args.shadow == 'CNN4':
            # Classifier: 4-layer CNN
            classifier_sub = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
            path_sub = 'Models/Classifier/CNN/Shadow/classifier.pth'
        elif args.shadow == 'ResNet':
            # Classifier: ResNet
            classifier_sub = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
            path_sub = 'Models/Classifier/ResNet/Shadow/classifier.pth'
    elif args.victim == 'ResNet':
        # Classifier: ResNet
        classifier = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
        path = 'Models/Classifier/ResNet/classifier.pth'
    
        if args.shadow == 'CNN4':
            # Classifier: 4-layer CNN
            classifier_sub = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
            path_sub = 'Models/Classifier/CNN/Shadow_1/classifier.pth'
        elif args.shadow == 'ResNet':
            # Classifier: ResNet
            classifier_sub = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
            path_sub = 'Models/Classifier/ResNet/Shadow_1/classifier.pth'

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
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch_sub, best_cl_acc_sub))
    logger.info("=> loaded substitute classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch_sub, best_cl_acc_sub))

    # Test the accuracy of the loaded classifier
    total = 0
    correct = 0

    classifier.eval()
    classifier_sub.eval()

    inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    # Train inversion model using complete structure
    best_recon_loss = 999
    early_stop_label = 0
    for epoch in range(1, args.epochs + 1):
        train(classifier, inversion, args.log_interval, device, train_loader, optimizer, epoch, args.penalty, args.norm, 'Inversion_Models/'+args.path_out, classifier_sub, args.lambda_pen, args.adv_param, logger, args.k_top, args.train_or_not, args.loss_mode)
        recon_loss = test(classifier, inversion, device, test_loader, epoch, 'test1', logger, 'Inversion_Models/'+args.path_out, pri_loader)
        # test(classifier, inversion, device, test2_loader, epoch, 'test2')

        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            torch.save(state, 'Inversion_Models/' + args.path_out + 'inversion.pth')
            shutil.copyfile('Inversion_Models/'+args.path_out + 'recon_round_private.png', 'Inversion_Models/' + args.path_out + 'best.png')
            # shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')

            early_stop_label = 0
        else:
            early_stop_label += 1
            if early_stop_label == args.early_stop:
                logger.info('\nThe best test inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, best_recon_loss))
                break

    
    

if __name__ == '__main__':
    main()