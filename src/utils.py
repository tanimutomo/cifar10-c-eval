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


def create_barplot(accs :dict, title :str, savepath :str):
    y = list(accs.values())
    x = np.arange(len(y))
    xticks = list(accs.keys())

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f'{j:.1f}', ha='center', va='bottom', fontsize=7)

    plt.title(title)
    plt.ylabel('Accuracy (%)')

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis='y')
    plt.savefig(savepath)
    plt.close()


def get_fname(weight_path :str):
    return '.'.join(weight_path.split('/')[-1].split('.')[:-1])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


if __name__ == '__main__':
    l = load_txt('./corruptions.txt')
    print(l)