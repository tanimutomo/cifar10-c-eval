import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]


def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1) # top-k index: size (B, k)
        pred = pred.t() # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)

        if len(acc) == 1:
            return acc[0]
        else:
            return acc


def create_barplot(accs :dict, savepath :str):
    y = list(accs.values())
    x = np.arange(len(y))
    xticks = list(accs.keys())

    plt.bar(x, y)

    plt.title(savepath)
    plt.ylabel('Accuracy (%)')

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks()

    plt.subplots_adjust(bottom=0.3)
    plt.savefig(savepath)


def get_fname(weight_path :str):
    return weight_path.split('/')[-1].split('.')[0]


if __name__ == '__main__':
    l = load_txt('./corruptions.txt')
    print(l)