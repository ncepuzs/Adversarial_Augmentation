from os import access
import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_acccuracy(classifier, test_dataloader, device, printable=0):
	total = 0
	correct = 0

	classifier.eval()
	with torch.no_grad():
		for i, (images, labels) in enumerate(test_dataloader):
			images, labels = images.to(device), labels.to(device)
			outputs = classifier(images)

			prob, pred_label = torch.max(outputs, dim=1)
			total += labels.size(0)
			correct += (pred_label == labels).sum().item()
	
	acc = correct * 1.0 / total
	if printable == 1:
		print("posi_num:{}, neg_num:{}".format(correct, total - correct))
		print("target model accuracy: ", acc)
	return acc

def compute_success(classifier, fake_data, labels, k_top, printable=0):
	total = 0
	fail = 0

	classifier.eval()
	with torch.no_grad():
		outputs = classifier(fake_data, release=True)
		# prob, pred_label = torch.max(outputs, dim=1)
		prob, pred_label = torch.topk(outputs, k_top, dim=1)

		total += labels.size(0)
		# fail += (pred_label == labels).sum().item()

		for i in range(pred_label.shape[0]):
			same_pred = [1 if x in pred_label[i] else 0 for x in labels[i]]
			if sum(same_pred) > 0:
				fail += 1
				
	suc = (total-fail) * 1.0 / total
	if printable == 1:
		print("posi_num:{}, neg_num:{}".format(fail, total-fail))
		print("target model accuracy: ", suc)
	return suc


def imshow(img, title):
	npimg = img.numpy()
	fig = plt.figure(figsize = (5, 15))
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.title(title)
	plt.show()