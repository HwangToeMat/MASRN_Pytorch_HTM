import torch

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if __name__ == '__main__':
    num = data_aug()
    print('Number of parameters is : ', num)
