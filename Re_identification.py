import argparse
from calendar import c
import logging
from unittest import TestLoader

from requests import PreparedRequest
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import os, shutil
from torchvision import transforms
from torchvision.transforms.functional import normalize
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from VGG import vgg11, vgg11_bn
import torch.nn.functional as F
import math


from model import Classifier, Inversion, FaceResNet, ResBlock

# Settings
parser = argparse.ArgumentParser(description='Re-identification for reconstructed images')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--path_out', type=str, default='Re_id_100_Black/')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--truncation', type=int, default=100)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=0, metavar='')
parser.add_argument('--pre_accuracy', type=float, default=0.9)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--penalty', type=str, default='no')
parser.add_argument('--norm', type=str, default='all')
parser.add_argument('--eps_adv', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--path_Inv', type=str, default='Inversion_Models_100_Black/wo_adv_wo_pen/')
parser.add_argument('--path_target', type=str, default='Models_100/wo_adv_wo_pen/')
parser.add_argument('--lambda_pen', type=float, default=0.0001)
parser.add_argument('--adv_param', type=str, default='(500, 0.1)')
parser.add_argument('--victim', type=str, default='CNN4')
parser.add_argument('--shadow', type=str, default='ResNet')



def main():
    args = parser.parse_args()

    # logging
    os.makedirs(args.path_out, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        filename=args.path_out + 're_id_loss.log',
                        filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认果不写的话，就是追加模式
                        format=
                        # '%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s'
                        '%(asctime)s -[line:%(lineno)d]: %(message)s'
                        # 日志格式
                        )
    logger = logging.getLogger(__name__)
    logger.info("\n")
    logger.info("================================")
    logger.info(args)
    # logger.info("eps_adv: {}, lambda: {}".format(args.eps_adv, args.lambda_pen))
    logger.info("adv_param: {}, lambda: {}".format(args.adv_param, args.lambda_pen))
    logger.info("lambda_pen: {}, lr: {}".format(args.lambda_pen, args.lr))

    logger.info("================================")
    print("")

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
    train_pri, test_pri = train_test_split(pri_set, test_size=0.2, random_state=42)
    print("len of train_data_classifier:", len(train_pri))
    print("len of test_data_classifier:", len(test_pri))
    logger.info("len of train_data_classifier:".format(len(train_pri)))
    logger.info("len of test_data_classifier:".format(len(test_pri)))

    test_loader = torch.utils.data.DataLoader(test_pri, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    '''
    Structure setting
    '''
    if args.victim == 'CNN4':
        # Classifier: 4-layer CNN
        classifier = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
        path = 'Models/Classifier/CNN/classifier.pth'
    elif args.victim == 'ResNet':
        # Classifier: ResNet
        classifier = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
        path = 'Models/Classifier/ResNet/classifier.pth'

    checkpoint = torch.load(path)
    classifier.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    logger.info("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))

    # checkpoint_target = torch.load(path_target)
    # classifier_target.load_state_dict(checkpoint_target['model'])
    # epoch = checkpoint_target['epoch']
    # best_cl_acc_target = checkpoint_target['best_cl_acc']
    # print("=> loaded classifier_target checkpoint '{}' (epoch {}, acc {:.4f})".format(path_target, epoch, best_cl_acc_target))
    # logger.info("=> loaded classifier_target checkpoint '{}' (epoch {}, acc {:.4f})".format(path_target, epoch, best_cl_acc_target))

    # Evaluation model: same architecture, different data (not trained)
    # path_eva_arc =  'Models_100/Inv_data/classifier100_0.99.pth'
    # classifier_eva_arc = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    # checkpoint_eva_arc = torch.load(path_eva_arc)
    # classifier_eva_arc.load_state_dict(checkpoint_eva_arc['model'])
    # epoch = checkpoint_eva_arc['epoch']
    # best_cl_acc_target = checkpoint_eva_arc['best_cl_acc']
    # print("=> loaded evaluation classifier with same architecture checkpoint '{}' (epoch {}, acc {:.4f})".format(path_target, epoch, best_cl_acc_target))
    # logger.info("=> loaded  evaluation classifier with same architecture checkpoint '{}' (epoch {}, acc {:.4f})".format(path_target, epoch, best_cl_acc_target))

    # Evaluation model: different architecture, same data (not trained)
    path_eva_resnet =  'Models/Classifier/ResNet/classifier.pth'
    # classifier_eva_data = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    classifier_eva_resnet = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
    checkpoint_eva_resnet = torch.load(path_eva_resnet)
    classifier_eva_resnet.load_state_dict(checkpoint_eva_resnet['model'])
    epoch = checkpoint_eva_resnet['epoch']
    best_cl_acc_target = checkpoint_eva_resnet['best_cl_acc']
    print("=> loaded evaluation classifier (ResNet) with same data checkpoint '{}' (epoch {}, acc {:.4f})".format(path_eva_resnet, epoch, best_cl_acc_target))
    logger.info("=> loaded  evaluation classifier (ResNet) with same data checkpoint '{}' (epoch {}, acc {:.4f})".format(path_eva_resnet, epoch, best_cl_acc_target))

    # Evaluation model: dCNN
    path_eva_cnn =  'Models/Classifier/CNN/classifier.pth'
    classifier_eva_cnn = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    # classifier_eva_re = nn.DataParallel(FaceResNet(ResBlock, args.nz)).to(device)
    checkpoint_eva_cnn = torch.load(path_eva_cnn)
    classifier_eva_cnn.load_state_dict(checkpoint_eva_cnn['model'])
    # classifier_eva_cnn.eval()
    epoch = checkpoint_eva_cnn['epoch']
    best_cl_acc_target = checkpoint_eva_cnn['best_cl_acc']
    print("=> loaded evaluation classifier (ResNet) with same data checkpoint '{}' (epoch {}, acc {:.4f})".format(path_eva_resnet, epoch, best_cl_acc_target))
    logger.info("=> loaded  evaluation classifier (ResNet) with same data checkpoint '{}' (epoch {}, acc {:.4f})".format(path_eva_resnet, epoch, best_cl_acc_target))

    # Evaluation model: general (VGG)
    path_eva_vgg =  'Models/Classifier/VGG/classifier.pth'
    # classifier_eva_data = nn.DataParallel(Classifier(nz=args.nz, nc=args.nc, ndf=args.ndf)).to(device)
    classifier_eva_vgg = nn.DataParallel(vgg11_bn(pretrained=False, num_classes=args.nz, init_weights=False)).to(device)
    checkpoint_eva_vgg = torch.load(path_eva_vgg)
    classifier_eva_vgg.load_state_dict(checkpoint_eva_vgg['model'])
    epoch = checkpoint_eva_vgg['epoch']
    best_cl_acc_vgg = checkpoint_eva_vgg['best_cl_acc']
    print("=> loaded evaluation classifier (VGG) with same data checkpoint '{}' (epoch {}, acc {:.4f})".format(path_eva_vgg, epoch, best_cl_acc_vgg))
    logger.info("=> loaded  evaluation classifier (VGG) with same data checkpoint '{}' (epoch {}, acc {:.4f})".format(path_eva_vgg, epoch, best_cl_acc_vgg))

    # Inversion
    inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)

    path = 'Inversion_Models/' + args.path_Inv + 'inversion.pth'
    
    checkpoint = torch.load(path)
    inversion.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_recon_loss = checkpoint['best_recon_loss']

    print("=> loaded inversion checkpoint '{}' (epoch {}, loss {:.4f})".format(path, epoch, best_recon_loss))
    logger.info("=> loaded inversion checkpoint '{}' (epoch {}, loss {:.4f})".format(path, epoch, best_recon_loss))

    # Inverse the test data and feed them into the classifier
    classifier.eval()
    inversion.eval()
    classifier_eva_resnet.eval()
    # classifier_eva_vgg.eval()
    plot = True
    correct = 0
    correct_ori = 0
    correct_target = 0
    correct_target_ori = 0
    correct_cnn = 0
    correct_resnet = 0
    correct_vgg = 0
    loss_recon = 0
    loss_PSNR = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Use Original classifier to re-identify and reconstruct images
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            pred = prediction.max(1, keepdim=True)[1]
            correct_ori += pred.eq(target.view_as(pred)).sum().item()
            reconstruction_raw = inversion(prediction)
            # print('data.shape:{}; targetshape:{}'.format(data.shape, target.shape))
            mse_loss = F.mse_loss(reconstruction_raw, data, reduction='mean')
            psnr_loss = 20 * torch.log10(255.0 / torch.sqrt(mse_loss))
            loss_recon += F.mse_loss(reconstruction_raw, data, reduction='sum').item()
            loss_PSNR += psnr_loss.item()

            reconstruction = normalize(reconstruction_raw, [0.5], [0.5])
            reconstruction = reconstruction_raw
            pre_prime = classifier(reconstruction, release=True)
            pred = pre_prime.max(1, keepdim=True)[1]
            # print("pre_prime.shape:{}; pred.shape:{}".format(pre_prime.shape, pred.shape))
            correct += pred.eq(target.view_as(pred)).sum().item()

            # # Use substitute classifier in training of inversion models to re-identify and reconstruct images
            # prediction_target = classifier_target(data, release=True)
            # pred_target = prediction_target.max(1, keepdim=True)[1]
            # correct_target_ori += pred_target.eq(target.view_as(pred_target)).sum().item()
            # reconstruction_target = inversion(prediction_target)
            # pre_target_prime = classifier_target(reconstruction_target, release=True)
            # pred = pre_target_prime.max(1, keepdim=True)[1]
            # # print("pre_prime.shape:{}; pred.shape:{}".format(pre_prime.shape, pred.shape))
            # correct_target += pred.eq(target.view_as(pred)).sum().item()

            # Evaluation on other evaluation models:
            # pre_arc = classifier_eva_arc(reconstruction, release=True)
            # pred_arc = pre_arc.max(1, keepdim=True)[1]
            # correct_arc += pred_arc.eq(target.view_as(pred_arc)).sum().item()

            pre_data = classifier_eva_resnet(reconstruction, release=True)
            pred_data = pre_data.max(1, keepdim=True)[1]
            correct_resnet += pred_data.eq(target.view_as(pred_data)).sum().item()

            pre_cnn = classifier_eva_cnn(reconstruction, release=True)
            pred_cnn = pre_cnn.max(1, keepdim=True)[1]
            correct_cnn += pred_cnn.eq(target.view_as(pred_cnn)).sum().item()

            pre_vgg = classifier_eva_vgg(reconstruction, release=True)
            pred_vgg = pre_vgg.max(1, keepdim=True)[1]
            correct_vgg += pred_vgg.eq(target.view_as(pred_vgg)).sum().item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 64:i * 64 + 32] = inverse[i * 32:i * 32 + 32]
                    out[i * 64 + 32:i * 64 + 64] = truth[i * 32:i * 32 + 32]
                vutils.save_image(out, args.path_out + 'recon.png', nrow=32, normalize=False)
                vutils.save_image(out, args.path_out + 'recon_norm.png', nrow=32, normalize=True)
                # save_imgs(out, num_in_row=32, row_num=32, path_out=args.path_out)
                plot = False

    loss =  loss_recon / len(test_loader.dataset) /(64*64)
    print("-------------------------------------------------------------")
    print("The reconstruction loss is :{}".format(loss))
    print("Compute PSNR based on MSE:{}".format(20 * math.log10(255.0 / math.sqrt(loss))))
    print("-------------------------------------------------------------")
    
    logger.info("-----------------------------------------------------------")
    logger.info("The reconstruction loss is :{}".format(loss))
    logger.info(("Compute PSNR based on MSE:{}".format(20 * math.log10(255.0 / math.sqrt(loss)))))
    logger.info("-----------------------------------------------------------")

    loss_PSNR_final =  loss_PSNR / len(test_loader.dataset) /(64*64)
    print("-------------------------------------------------------------")
    print("The PSNR loss is :{}".format(loss_PSNR_final))
    print("-------------------------------------------------------------")
    
    logger.info("-----------------------------------------------------------")
    logger.info("The PSNR loss is :{}".format(loss_PSNR_final))
    logger.info("-----------------------------------------------------------")

    accu =  correct / len(test_loader.dataset)
    accu_ori = correct_ori / len(test_loader.dataset)
    # print("The number of correctly-classified examples:{}, total number is {}".format(correct_ori, len(test_loader.dataset)))
    # print("The attack accuracy by original classifier:{}".format(accu))
    print("-------------------------------------------------------------")
    
    logger.info("=============The accuracy on original images is: {}.".format(accu_ori))
    logger.info("=============The accuracy on reconstructed images is:{}.".format(accu))
    logger.info("-----------------------------------------------------------")

    # Evaluation : same architecture
    # accu_arc =  correct_arc / len(test_loader.dataset)
    # print("The number of correctly-classifier examples by classifier_arc:{}, total number is {}, acc:{}/{}".format(correct_arc, len(test_loader.dataset), accu_arc, best_cl_acc))
    # print("-------------------------------------------------------------")
    
    # logger.info("\t The accuracy on reconstructed images by classifier_arc:{}.".format(accu_arc))
    # logger.info("-----------------------------------------------------------")

    # Evaluation : same data
    accu_data =  correct_resnet / len(test_loader.dataset)
    print("The number of correctly-classifier examples by classifier_resnet:{}, total number is {}, acc:{}/{}".format(correct_resnet, len(test_loader.dataset), accu_data, best_cl_acc))
    print("-------------------------------------------------------------")
    
    logger.info("\t The accuracy on reconstructed images by classifier_resnet::{}.".format(accu_data))
    logger.info("-----------------------------------------------------------")

    # Evaluation : same data
    accu_cnn =  correct_cnn / len(test_loader.dataset)
    print("The number of correctly-classifier examples by classifier_cnn:{}, total number is {}, acc:{}/{}".format(correct_cnn, len(test_loader.dataset), accu_cnn, best_cl_acc))
    print("-------------------------------------------------------------")
    
    logger.info("\t The accuracy on reconstructed images by classifier_cnn::{}.".format(accu_cnn))
    logger.info("-----------------------------------------------------------")

    # Evaluation : vgg
    accu_vgg =  correct_vgg / len(test_loader.dataset)
    print("The number of correctly-classifier examples by classifier_vgg:{}, total number is {}, acc:{}/{}".format(correct_vgg, len(test_loader.dataset), accu_vgg, best_cl_acc))
    print("-------------------------------------------------------------")
    
    logger.info("\n ============ The accuracy on reconstructed images by classifier_vgg :{}.".format(accu_vgg))
    logger.info("-----------------------------------------------------------")
    

    # accu_subtitute =  correct_target / len(test_loader.dataset)
    # print("The number of correctly-classifier examples:{}, total number is {}, acc:{}/{}".format(correct_target, len(test_loader.dataset), accu_subtitute, best_cl_acc_target))
    # print("The number of correctly-classifier examples by original:{}, total number is {}".format(correct_target_ori, len(test_loader.dataset)))
    
    # logger.info("The accuracy on original images is: {}.".format(best_cl_acc_target))
    # logger.info("\t The accuracy on reconstructed images is:{}.".format(accu_subtitute))
                

def save_imgs(tensor_vector, num_in_row, row_num, path_out):
    for row_i in range(1, row_num + 1):
        start_i = (row_i-1) * num_in_row
        end_i = row_i * num_in_row
        data_i = tensor_vector[start_i: end_i]
        for item_i in range(0, len(data_i)):
            img_PIL = np.array(data_i[item_i][0].cpu(), dtype='float32')
            plt.subplot(row_num, len(data_i), start_i+item_i+1)
            plt.imshow(img_PIL, cmap='gray')
            plt.axis('off')
    # plt.subplots_adjust(wspace=-0.05, hspace=0)
    plt.savefig(path_out + 'recon_plt.png', dpi=4096, bbox_inches='tight')

if __name__ == '__main__':
    main()