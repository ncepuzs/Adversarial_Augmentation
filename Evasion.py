from platform import release
import torch
import torch.nn.functional as F
from tools import compute_success
import torchvision.utils as vutils
import os
import numpy as np
"""
Untarget Projected Gradient Decent Attack
:param classifier: The victim model
:param imgs: Inputs to perturb
:param labels: The original label
:param norm: The order of norm of perturbations
:param alpha: The step size of perturbation
:param eps: The maximal perturbation
:param iters: The maximal steps for attacks
"""

def get_gpu_memory():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')
    return memory_gpu


def PGD(classifier, imgs, labels, device, norm, eps=0.4, alpha=2/255, iters=40):
    if norm not in [1, 2, "inf"]:
        raise ValueError('Norm must be either 1, 2 or inf')

    imgs = imgs.to(device)
    labels = labels.to(device)
    imgs_adv = imgs.detach().clone()

    classifier.eval()
    for i in range(iters):
        imgs_adv.requires_grad = True
        output = classifier(imgs_adv)

        classifier.zero_grad()
        loss = F.nll_loss(output, labels)
        loss.backward()
        grad = imgs_adv.grad
        
        grad = select_norm(norm, grad)

        imgs_adv = imgs_adv + alpha * grad

        # projection
        perturbation = imgs_adv - imgs
        imgs_adv = imgs + torch.clamp(perturbation, min=-eps, max=eps)
        imgs_adv = torch.clamp(imgs_adv, min=0, max=1).detach_()

    output = classifier(imgs_adv)
    prob, pred_label = torch.max(output, dim=1)
        # print("debug: success once")

    # Crafting adversarial examples in batch cannot apply a end contdition.

    return (imgs_adv, pred_label, labels)
    
def PGD_targeted(classifier, imgs, labels, device, norm, path_out, eps=0.4, alpha=2/255, iters=40):
    if norm not in [1, 2, "inf"]:
        raise ValueError('Norm must be either 1, 2 or inf')

    imgs = imgs.to(device)
    labels = labels.to(device)
    imgs_adv = imgs.detach().clone()

    classifier.eval()
    for i in range(iters):
        imgs_adv.requires_grad = True
        output = classifier(imgs_adv)

        classifier.zero_grad()
        loss = F.nll_loss(output, labels)
        loss.backward()
        grad = imgs_adv.grad
        
        grad = select_norm(norm, grad)

        imgs_adv = imgs_adv - alpha * grad

        # projection
        perturbation = imgs_adv - imgs
        imgs_adv = imgs + torch.clamp(perturbation, min=-eps, max=eps)
        imgs_adv = torch.clamp(imgs_adv, min=0, max=1).detach_()
        # print("debug: success once")

    #compute success
    fail = compute_success(classifier,  imgs_adv, labels, 0)
    print("===--=== The success rate of targeted PGD is {}".format(1-fail))
    truth = imgs[0:32]
    adv = imgs_adv[0:32]
    out = torch.cat((adv, truth))
    try:
        for i in range(4):
            out[i * 16:i * 16 + 8] = adv[i * 8:i * 8 + 8]
            out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
        vutils.save_image(out, path_out + 'Crafted_adv_examples.png', nrow=8, normalize=True)
    except:
        print("Incrrectly-classified is not enough. The imgs_adv.shape is {}".format(imgs_adv.shape))
    finally:
        print("Continue to run.") 
    # Crafting adversarial examples in batch cannot apply a end contdition.

    return imgs_adv


def select_norm(norm, grad):
    avoid_0 = 10**(-8)
    if norm == 1:
        ind = tuple(range(1, len(grad.shape)))
        grad = grad / (torch.sum(grad.abs(), dim=ind, keepdim=True) + avoid_0)

    elif norm == 2:
        ind = tuple(range(1, len(grad.shape)))
        grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + avoid_0)  # type: ignore

    elif norm == "inf":
        grad = grad.sign()

    return grad

'''
Copy from https://github.com/cg563/simple-blackbox-attack/blob/418b25f4dd8cc0f988376c5730ae3988b95fbce0/simba.py#L6
Simple balck-box adversarial attacks

Use SimBA class:
    attacker = SimBA(classifier, 'FaceScrub', 64)
    # no requirement of max num_runs
    for i in range(N):
        if not args.targeted:
            adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
            images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, linf_bound=args.linf_bound,
            order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)

'''

class SimBA:
    
    def __init__(self, model, dataset, image_size, pseudo, k_top):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()
        self.pseudo = pseudo
        self.k = k_top
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        # x = x.view(-1, 3, size, size)
        # z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        x = x.view(-1, 1, size, size)
        # z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
        
    # def normalize(self, x):
    #     return utils.apply_normalization(x, self.dataset)

    def get_probs(self, x, y):
        # output = self.model(self.normalize(x.cuda())).cpu()
        output = self.model(x.cuda(), release=True).cpu()
        # probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        if self.pseudo == False:
            probs = torch.index_select(output, 1, y)
        else:
            # y is a pseudo label （ length=k )
            # yi indicates the i-th label in the pseudo label
            probs = torch.zeros(x.shape[0])
            for i in range(self.k):
                exec('y{} = torch.index_select(y, 1, torch.tensor([{}])).view(-1)'.format(i, i))
                exec('prob{} = torch.diag(torch.index_select(output, 1, y{}))'.format(i, i))
                exec('probs += prob{}'.format(i))

        return probs
    
    def get_preds(self, x):
        # output = self.model(self.normalize(x.cuda())).cpu()
        output = self.model(x.cuda(), release=True).cpu()
        # _, preds = output.data.max(1)

        # Pseudo label = False
        # _, preds = output.max(1)
        # preds = preds.view(-1)

        # Pseudo label = True
        _, preds = torch.topk(output, self.k, dim=1)
        return preds
    
    def pred_equal(self, x1, x2):

        equal_ts = torch.ones(x1.shape[0]).eq(torch.ones(x1.shape[0]))

        for i in range(equal_ts.shape[0]):
            same_pred = [1 if x in x1[i] else 0 for x in x2[i]]
            if sum(same_pred) == 0:
                equal_ts[i] = False
                # print("pred_equal() == False:\n x1={},\nx2={}".format(x1[i],x2[i]))

        return equal_ts

    def pred_ne(self, x1, x2):

        ne_ts = torch.ones(x1.shape[0]).eq(torch.ones(x1.shape[0]))

        for i in range(ne_ts.shape[0]):
            same_pred = [1 if x in x1[i] else 0 for x in x2[i]]
            if sum(same_pred) > 0:
                ne_ts[i] = False

        return ne_ts


    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

    def block_order(self, image_size, channels, initial_size=1, stride=1):
        order = torch.zeros(channels, image_size, image_size)
        total_elems = channels * initial_size * initial_size
        perm = torch.randperm(total_elems)
        order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
        for i in range(initial_size, image_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = torch.randperm(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
            order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
            total_elems += num_elems
        return order.view(1, -1).squeeze().long().sort()[1]

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def simba_batch(self, images_batch, labels_batch, max_iters, epsilon, linf_bound=0.0,
                    targeted=False, log_every=1):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        # device = torch.device("cuda")
        images_batch = images_batch.cpu()
        labels_batch = labels_batch.cpu()

        clamp_inf = 0

        # sample a random ordering for coordinates independently per batch element
        # if order == 'rand':
        #     indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        # elif order == 'diag':
        #     indices = utils.diagonal_order(image_size, 3)[:max_iters]
        # elif order == 'strided':
        #     indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        # else:
        #     indices = utils.block_order(image_size, 3)[:max_iters]
        indices = self.block_order(image_size, 1)[:max_iters]
        # if order == 'rand':
        #     expand_dims = freq_dims
        # else:
        #     expand_dims = image_size
        expand_dims = image_size
        # n_dims = 3 * expand_dims * expand_dims
        n_dims = 1 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        # if pixel_attack:
        #     trans = lambda z: z
        # else:
        #     trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        # trans = lambda z: z
        remaining_indices = torch.arange(0, batch_size).long()
        
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + x[remaining_indices].view(-1, 1, expand_dims, expand_dims)).clamp(clamp_inf, 1)
            
            ''' debug'''
            # print("c(img):", self.model(images_batch[0], release=True))
            # print("c(exp):", self.model(expanded[0], release=True))
            # print("img[0]==exp[0]", images_batch[0]==expanded[0])
            # print("img[0]:", images_batch[0][0][0])
            # print("exp[0]:", expanded[0][0][0])
            # print("x is ", x[0])
            # input("Compare")

            # perturbation = trans(self.expand_vector(x, expand_dims))
            perturbation = x.view(-1, 1, expand_dims, expand_dims)
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            # print("expanded.shape:", expanded.shape)
            # print("preds_next.shape:", preds_next.shape)
            preds[remaining_indices] = preds_next
            if targeted:
                # remaining = preds.ne(labels_batch)
                remaining = self.pred_ne(preds, labels_batch)
            else:
                remaining = self.pred_equal(preds, labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                # adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                adv = (images_batch + x.view(-1, 1, expand_dims, expand_dims)).clamp(clamp_inf, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k-1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            # adv = (images_batch[remaining_indices].to(device) + trans(self.expand_vector(left_vec, expand_dims)).to(device)).clamp(0, 1)
            adv = (images_batch[remaining_indices] + left_vec.view(-1, 1, expand_dims, expand_dims)).clamp(clamp_inf, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            # adv = (images_batch[remaining_indices].to(device) + trans(self.expand_vector(right_vec, expand_dims)).to(device)).clamp(0, 1)
            adv = (images_batch[remaining_indices] + right_vec.view(-1, 1, expand_dims, expand_dims)).clamp(clamp_inf, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
                # gpu_memory = get_gpu_memory()
                # gpu_list = np.argsort(gpu_memory)[::-1]
                # gpu_list_str = ','.join(map(str, gpu_list))
                # os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

                # print("The rest of Memory:{}".format(torch.cuda.device_count))
                # print("gpu_free_memory:{}".format(gpu_memory))
                # print("CUDA_VISIBLE_DEVICES:{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
            
            # if x.sum() > 0.001:
            #     print("\nx is \n{}".format(x))
            #     input("x > 0.001")
            # else:
            #     input("x <= 0.001")
        # expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        expanded = (images_batch + x.view(-1, 1, expand_dims, expand_dims)).clamp(clamp_inf, 1)

        preds = self.get_preds(expanded)
        if targeted:
            # remaining = preds.ne(labels_batch)
            remaining = self.pred_ne(preds, labels_batch)
        else:
            # remaining = preds.eq(labels_batch)
            remaining = self.pred_equal(preds, labels_batch)

        succs[:, max_iters-1] = ~remaining
        # print("------------------\n ~remaining is :", ~remaining)
        # return expanded[~remaining], preds[~remaining], labels_batch[~remaining], probs, succs, queries, l2_norms, linf_norms
        return expanded, preds, labels_batch, probs, succs, queries, l2_norms, linf_norms
