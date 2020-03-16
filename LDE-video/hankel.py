import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from matplotlib import *
from pylab import *
import time

def hankel_singular_values(H, compute_uv = True, gpu_id=0):
    sv = []#
    for n in range(H.size(0)):
        sv.append(torch.svd(H[n,:])[1].unsqueeze(0))
    return torch.cat(sv, dim=0)

def represent_sv_heatmap(sv, name):
    '''
    :param sv: n_dim, n_sv
    :return:
    '''
    dir = ''
    plt.figure()
    svM = torch.max(sv)
    # svM[svM == 0] = 1
    sv = sv.div(svM)
    plt.pcolormesh(torch.t(sv).data.cpu().detach().numpy(), cmap='Reds')
    plt.colorbar()
    plt.savefig(os.path.join(dir, name + '.png'), dpi=200)  # , dpi=200
    # plt.show()
    plt.close()

    # print(np.mean(sv.data.cpu().detach().numpy(), axis=0))
    return

def sv_percentage_mean(sv):
    '''
    :param sv: n_dim, n_sv
    :return:
    '''
    dir = '../figures'
    # plt.figure()
    svsum = torch.sum(sv, dim=1).unsqueeze(1)
    svsum[svsum == 0] = 1
    sv = sv.div(svsum)
    sv = torch.mean(sv, dim=0)
    sv_cpu = sv.data.cpu().detach().numpy()

    return sv_cpu

def gram_nuclear_norm(y, gid):
    '''y shape:
    [m+p, n_features, T]'''
    wsize = int((y.shape[2] + 1) / 2)
    ksize = [1, wsize]
    poolL2  = nn.LPPool2d(2, ksize, stride=1).cuda(gid) # ceil_mode?
    avgpool = nn.AvgPool2d(ksize, stride=1).cuda(gid)
    gnn = avgpool(poolL2(y)**2) # * wsize

    return gnn

def unbias_seq(ts):
    mean = torch.mean(ts, dim=2)
    ts = ts - mean.unsqueeze(2)
    return ts

def hankel_matrix(y, gpu_id=None, unbiased = False, diff = False):

    '''y shape:
    [m+p, n_features, T]'''

    '''Create Hankel matrix'''

    if diff:
        y = y[:, :, 1:] - y[:, :, :-1]

    if unbiased:
        y = unbias_seq(y)

    '''Square case:'''
    assert y.shape[2]%2 == 1
    sz = int((y.shape[2] + 1) / 2)


    chan = y.shape[0]
    nfeat = y.shape[1]
    nr = sz
    nc = sz
    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    Hidex = p1.add(p2)
    reHidex = Hidex.view(1,-1).type(torch.LongTensor)
    # reHidex = reHidex.numpy()

    T = y.shape[2]
    y = y.contiguous().view(-1,T)
    # H = y[:, :, reHidex].view(chan, T, nr, nc)
    H = y[:, reHidex].view(-1, nr, nc)

    return H

def gram_matrix(y, delta=0, gpu_id=0, unbiased = False, diff = False):

    '''y shape:
    [m+p, n_features, T]'''

    '''Create Hankel matrix'''

    if diff:
        y = y[:, :, 1:] - y[:, :, :-1]

    if unbiased:
        y = unbias_seq(y)

    '''Square case:'''
    assert y.shape[2]%2 == 1
    sz = int((y.shape[2] + 1) / 2)


    chan = y.shape[0]
    nfeat = y.shape[1]
    nr = sz
    nc = sz
    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    Hidex = p1.add(p2)
    reHidex = Hidex.view(1,-1).type(torch.LongTensor)
    # reHidex = reHidex.numpy()

    T = y.shape[2]
    y = y.contiguous().view(-1,T)
    # H = y[:, :, reHidex].view(chan, T, nr, nc)
    H = y[:, reHidex].view(-1, nr, nc)
    G = torch.matmul(H, H.permute(0,2,1))
    if delta!=0:
        G = G + delta*torch.eye(nr, nc).cuda(gpu_id)
    # G_vec = G.view(1, -1)

    return G

def comp_gram_matrix(y_ini, delta=0, gpu_id=0, unbiased = False, diff = False, pairwise=None):

    '''y shape:
    [dim_z, n_features, T]'''

    '''Create Hankel matrix'''

    y = y_ini

    if diff:
        y = y[:, :, 1:] - y[:, :, :-1]

    if unbiased:
        y = unbias_seq(y)

    '''Square case:'''
    assert y.shape[2]%2 == 1
    sz = int((y.shape[2] + 1) / 2)


    chan = y.shape[0]
    nfeat = y.shape[1]
    nr = sz
    nc = sz
    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    Hidex = p1.add(p2)
    reHidex = Hidex.view(1,-1).type(torch.LongTensor)

    H = y[:, :, reHidex].view(chan, nfeat, nr, nc).permute(1,0,2,3)\
        .contiguous().view(nfeat, chan * nr, nc) #.view(-1, nr, nc)

    G = torch.matmul(H.permute(0, 2, 1), H)

    if y.size(1) == 1:
        normH = torch.norm(G.squeeze()).pow(.5)
        if normH == 0:
            normH = 1
        H_norm = H / normH
        G_norm = torch.matmul(H_norm.permute(0,2,1), H_norm)

    if pairwise=='inter':
        y_int = comb_pairwise_distance(y_ini)
        H_int = y_int[:, :, reHidex].view(chan, nfeat**2, nr, nc).permute(1,0,2,3)\
            .contiguous().view(nfeat**2, chan * nr, nc) # pairwise comb distance
        G_int = torch.matmul(H_int.permute(0,2,1), H_int)
        #It's performing still compound Hankel matrix with all pairwise dist
    elif pairwise=='intra':
        y_int = comb_pairwise_distance(y_ini.permute(1,0,2))
        H_int = y_int[:, :, reHidex].view(nfeat, chan ** 2, nr, nc)\
            .contiguous().view(nfeat, (chan**2) * nr, nc)
        G_int = torch.matmul(H_int.permute(0, 2, 1), H_int)
    else:
        if y.size(1) == 1:
            G_int = G_norm
        else:
            G_int = None

    if delta!=0:
        G = G + delta * torch.eye(nc, nc).cuda(gpu_id)
        if pairwise=='inter' or pairwise=='intra':
            G_int = G_int + delta * torch.eye(nc, nc).cuda(gpu_id)

    # class NuclearLossFunc(nn.Module):
    #     def __init__(self):
    #         super(NuclearLossFunc, self).__init__()
    #         return
    #
    #     def forward(self, mat):
    #         loss = torch.zeros([1]).cuda()
    #         total_batch, total_channel, Sx, Sy = mat.size()
    #         mat = mat.view(-1, Sx, Sy)
    #         mat_trans = torch.transpose(mat, 1, 2)
    #         m_total = torch.bmm(mat_trans, mat)
    #         loss = m_total.sum(0).trace()
    #         loss /= (total_batch * total_channel)  # Review this line
    #         return loss

    return G, H

def comb_pairwise_distance(y):

    '''y shape:
    [dim_z, n_features, T]'''

    dim_z = y.size(0)
    n_feat = y.size(1)
    T = y.size(2)

    y = y.permute(1,0,2)
    Y = y.expand(n_feat, n_feat, dim_z, T)
    Yt = Y.permute(2,1,0,3).contiguous().view(dim_z, n_feat**2, T)
    Y = Y.contiguous().view(n_feat**2, dim_z, T).permute(1,0,2)
    y_diff = (Y - Yt)

    # a=y_diff
    # print(y, '\n', a.shape, '\n', a, '\n', Y, '\n', Yt)

    return y_diff

def reweighted_loss(G_old, G, gpu_id=0):

    Ginv_old = torch.inverse(G_old)

    '''Norm'''
    Ginv_old_vec = Ginv_old.contiguous().view(Ginv_old.shape[0], -1)
    Ginv_old_norm = Ginv_old_vec / torch.sum(torch.norm(Ginv_old_vec, p=2, dim=1))
    # Ginv_old_norm = Ginv_old  # back to normal

    Ginv_old_norm_vec = Ginv_old_norm.contiguous().view(-1, 1).squeeze()#.cuda(gid))
    G_vec = G.view(-1, 1).squeeze()
    rw_gnn = torch.dot(G_vec,Ginv_old_norm_vec)

    return rw_gnn

if __name__ == "__main__":

    gpu_id = 0
    # y = torch.arange(27).float().unsqueeze(0)
    y = Variable(torch.from_numpy(np.random.randint(5, size=(4, 3, 9))))\
        .type('torch.FloatTensor').cuda(gpu_id)
    # y = Variable(torch.from_numpy(np.array([[0, 1, 2],[3, 4, 6], [7, 8, 9]])))\
    #     .type('torch.FloatTensor').cuda(gpu_id)
    y_old = Variable(torch.from_numpy(np.random.randint(5, size=(2, 3, 5))))\
        .type('torch.FloatTensor').cuda(gpu_id)


    # y = y.view(2,3,-1).permute(1,2,0)
    # Y = y.expand(y.size(0), y.size(0), y.size(1), y.size(2))
    # Yt = Y.permute(1,0,2,3).contiguous().view(9,-1,2).permute(2,0,1)
    # Y = Y.contiguous().view(9,-1,2).permute(2,0,1)
    # # a = pd(Y.contiguous().view(9,-1), Yt.contiguous().view(9,-1))
    # a = Y - Yt

    # a = comb_pairwise_distance(y)

    # Hsize = torch.Size([14, 14])
    # gnn = gram_nuclear_norm(y, gpu_id)
    # hnn = hankel_nuclear_norm(y, Hsize, gpu_id)
    # G = comp_gram_matrix(y_old, delta = 0.001)

    # Re-weighted heuristic
    # G_old = gram_matrix(y_old, delta = 0.001)
    # G = gram_matrix(y)
    # G_old = torch.eye(3).unsqueeze(0).repeat(2, 1, 1).cuda(0)
    # rwl = reweighted_loss(G_old, G)
    H = hankel_matrix(y)
    sv = hankel_singular_values(H)
    # represent_sv_heatmap(sv, 'ranks')
    sv = sv_percentage_mean(sv)
    # G_old = torch.eye(4).unsqueeze(0).repeat(6,1,1)

    print(y, '\n', H, '\n', H.shape, '\n', sv.shape, '\n', sv)
    # print(G_old.shape, '\n', G_old, '\n', rwl,'\n', y)

# def hankel_nuclear_norm(y, gpu_id):
#
#     '''y shape:
#     [m+p, n_features, T]'''
#
#     '''Create Hankel matrix'''
#
#     '''Square case:'''
#     assert y.shape[2]%2 == 1
#     sz = int((y.shape[2]+1)/2)
#
#     chan = y.shape[0]
#     nfeat = y.shape[1]
#     nr = sz
#     nc = sz
#     sv = Variable(torch.zeros(y.shape[0]*y.shape[1])).cuda(gpu_id)
#     ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
#     cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)
#
#     p1 = ridex*torch.ones([nr, nc])
#     p2 = cidex.expand(nr, nc)
#     Hidex = p1.add(p2)
#     reHidex = Hidex.view(1,-1).type(torch.LongTensor)
#     # reHidex = reHidex.numpy()
#
#     T = y.shape[2]
#     y = y.contiguous().view(-1,T)
#     # H = y[:, :, reHidex].view(chan, T, nr, nc)
#     H = y[:, reHidex].view(-1, nr, nc)
#
#     '''indexing kills the grad?'''
#     # t0 = time.time()
#     for n in range(chan * nfeat):
#
#         sv[n]=(torch.sum(torch.svd(H[n,:])[1]))
#         # sv[n] = (torch.sum(torch.eig(H[n, :], eigenvectors=False)[0]))
#         # print(H[n, :, :].shape)
#         # sv[n] = torch.inverse(H[n, :, :])
#
#     # sv = torch.inverse((H))
#     '''If we want to do the nn of Gram matrix:
#     2normpooling with 1x((T+1)/2) filter to y --> suma per files
#     maybe finetune with cardinality minimization'''
#
#     # print('Total time: '+str((time.time()-t0)))
#     return torch.mean(sv)
