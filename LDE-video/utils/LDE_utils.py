import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
# import factorization

# Note: Factorize and construct matrix K

def get_K(M):
    # assert len(M.size()) ==
    return torch.bmm(M,M.permute(0, 2, 1))

def factorize(K):
    F = None
    return F

def get_G(M):
    '''
    M: [*, T, f] --  T = n_frames_input
    G: [*, T/2 -1, T/2 -1] -- T/2 -1 = autoregressor_size
    '''

    size_G = M.size(1) // 2 + M.size(1) % 2
    G = torch.zeros(M.size(0), size_G, size_G).cuda()
    # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)

    for i in range(M.size(1) - size_G + 1):
        a = M[:, i:i+size_G]

        G += torch.bmm(a, a.permute(0, 2, 1)) #- meanK # With unbias

    return G

def get_partial_G(M, L=0):
    '''
    M: [*, T, f] --  T = n_frames_input
    G: [*, T/2 -1, T/2 -1] -- T/2 -1 = autoregressor_size
    '''

    Gs = []
    # size_G = M.size(1) // 2 + M.size(1) % 2
    num_g = M.size(1) - 2*L + 1
    for idx_g in range(num_g):
        G = torch.zeros(M.size(0), L, L).cuda()
        for i in range(L):
            # if idx_g == 26:
            #     print('a')
            a = M[:, idx_g+i:idx_g+i+L]
            G += torch.bmm(a, a.permute(0, 2, 1))
        Gs.append(G)
    Gs = torch.cat(Gs, dim=0)
    return Gs

def get_trace_K(M, flag='k'):

    '''
    :param M: [*, T, f] --  T = n_frames_input
    :return: scalar
    '''
    if flag == 'k':

        vecM = M.contiguous().view(M.size(0), 1, -1)

        # With unbias
        # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # tr = (vecM**2 - meanK).sum(dim=2, keepdim=True)

        # Without unbias
        tr = torch.bmm(vecM, vecM.permute(0, 2, 1))

    elif flag == 'g':
        tr = 0
        size_G = M.size(1) // 2 + M.size(1) % 2
        # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)

        for i in range(M.size(1) - size_G + 1):
            a = M[:, i:i + size_G]
            veca = a.contiguous().view(a.size(0), 1, -1)

            # With unbias
            # tr += (veca ** 2 - meanK).sum(dim=2, keepdim=True)

            # Without unbias
            tr += torch.bmm(veca, veca.permute(0, 2, 1))

    else:
        print('Wrong flag')
        tr = None

    return tr

def get_dist(M, neigh):
    #
    # dist = []
    # # TODO: check in GPU
    # print(M.shape)
    # for i in range(M.size(1)):
    #     dist.append(torch.cdist(M[0:1, i], M[:, neigh[0:1, i]], p=2))
    #
    # dist1 = torch.cat(dist, dim=1).squeeze(2)

    K = get_K(M) #Note: erase cuda()

    i_index = torch.arange(0,M.size(1)).repeat(M.size(0),1,1).permute(0,2,1)
    # print(i_index.dtype)
    # print(neigh.dtype)
    neigh = neigh.long()
    neigh = torch.cat((i_index,neigh),2)
    i = neigh[:,:,0:1].cuda()
    j = neigh[:,:,1:].cuda()

    Kii = torch.gather(K,2,i)
    Kij = torch.gather(K,2,j)

    Kjj = []
    for b in range(M.size(0)):
        Kjj.append(K[b,j[b,:],j[b,:]].unsqueeze(0))

    Kjj = torch.cat(Kjj,0)


    dist = Kii + Kjj - 2*Kij

    # print(K, '\n', Kii, '\n', Kjj, '\n',Kij, '\n',)
    return dist

# def normalize():
#     """
#     """
#     torch.max()


def main():

    n_frames_input = 6
    n = n_frames_input // 2 + n_frames_input % 2
    neigh = torch.randint(6,(1,6,2))

    '''Test get_G, get_K'''
    M = torch.Tensor([np.linspace(11, 5, 7), np.linspace(4, 10, 7)]).unsqueeze(0).permute(0,2,1).float()
    Gs = get_partial_G(M, 2).squeeze()
    # K = get_K(M).squeeze()
    # trK = torch.trace(K)
    # trK2 = get_trace_K(M, 'k')
    # trG = torch.trace(Gs)
    # trG2 = get_trace_K(M, 'g')
    # dist = get_dist(M, neigh)

    # print(M.shape)
    # print(Gs.shape)

    # print(G.size(), M.size(), '\n',M,'\n', G,'\n',K)
    # test

if __name__ == "__main__":
    main()