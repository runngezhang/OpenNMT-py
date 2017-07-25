import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.autograd import gradcheck
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


tmp_ = torch.rand(1,1).cuda()

KNN_CODE = """
extern "C" {

    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }

    __global__ void knn_fwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ h, float * __restrict__ c,
                            const int use_tanh)
    {
        assert ((k == 3) || (x == NULL));

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);

        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;

        for (int row = 0; row < len; ++row)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = use_tanh ? tanh(cur) : cur;
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u;
            xp += ncols_x;
            cp += ncols;
            hp += ncols;
        }
    }

    __global__ void knn_bwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h, const float * __restrict__ c,
                            const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ grad_u, float * __restrict__ grad_x,
                            float * __restrict__ grad_bias, float * __restrict__ grad_init,
                            int use_tanh)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *xp = (k == 3) ? (x + col + (len-1)*ncols) : (up + 3);
        const float *cp = c + col + (len-1)*ncols;

        const float *ghp = grad_h + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float *gxp = (k == 3) ? (grad_x + col + (len-1)*ncols) : (gup + 3);

        for (int row = len-1; row >= 0; --row)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);

            const float c_val = use_tanh ? tanh(*cp) : (*cp);
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));

            const float gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0

            // grad wrt x
            *gxp = gh_val*(1-g2);

            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;

            // grad wrt c
            const float tmp = use_tanh ? (g2*(1-c_val*c_val)) : g2;
            const float gc = gh_val*mask*tmp + cur;

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols_u;
            xp -= ncols_x;
            cp -= ncols;
            gup -= ncols_u;
            gxp -= ncols_x;
            ghp -= ncols;
        }

        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }

}
"""

KNN_PROG = Program(KNN_CODE, 'knn_prog.cu')
KNN_PTX = KNN_PROG.compile()
KNN_MOD = function.Module()
KNN_MOD.load(bytes(KNN_PTX.encode()))
KNN_FWD_FUNC = KNN_MOD.get_function('knn_fwd')
KNN_BWD_FUNC = KNN_MOD.get_function('knn_bwd')

Stream = namedtuple('Stream', ['ptr'])
KNN_STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)


class KNN_Compute(Function):

    def __init__(self, use_tanh, d_out):
        super(KNN_Compute, self).__init__()
        self.use_tanh = use_tanh
        self.d_out = d_out

    def forward(self, u, x, bias, init=None, mask_h=None):
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) / d
        ncols = batch*d
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)/thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d) if x.dim() == 3 else (batch, d)
        c = x.new(*size)
        h = x.new(*size)

        KNN_FWD_FUNC(args=[
            u.data_ptr(),
            x.data_ptr() if k == 3 else 0,
            bias.data_ptr(),
            init_.data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            length,
            batch,
            d,
            k,
            h.data_ptr(),
            c.data_ptr(),
            self.use_tanh],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=KNN_STREAM
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c

        return h, c[-1] if x.dim() == 3 else c

    def backward(self, grad_h, grad_last):
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) / d
        ncols = batch*d
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)/thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d)
        grad_init = x.new(batch, d)
        size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        #grad_x = x.new(*x.size()) if k == 3 else x.new(*size).zero_()
        grad_x = x.new(*x.size()) if k == 3 else None

        KNN_BWD_FUNC(args=[
            u.data_ptr(),
            x.data_ptr() if k == 3 else 0,
            bias.data_ptr(),
            init_.data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.data_ptr(),
            grad_last.data_ptr(),
            length,
            batch,
            d,
            k,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.use_tanh],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=KNN_STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None



class FastKNNCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0, use_tanh=0):
        super(FastKNNCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.use_tanh = use_tanh

        if n_in != n_out:
            self.weight = nn.Parameter(torch.Tensor(n_in, n_out, 4))
        else:
            self.weight = nn.Parameter(torch.Tensor(n_in, n_out, 3))
        self.bias = nn.Parameter(torch.Tensor(n_out*2))

    def forward(self, input, c0):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.view(-1, n_in)
        u = x_2d.mm(self.weight.view(n_in, -1))

        if self.training and (self.dropout>0):
            mask_h = self.get_dropout_mask_((batch, n_out), self.dropout)
            h, c = KNN_Compute(self.use_tanh, n_out)(u, input, self.bias, c0, mask_h)
        else:
            h, c = KNN_Compute(self.use_tanh, n_out)(u, input, self.bias, c0)

        #h, c = KNN_Compute(self.use_tanh, n_out)(u, input, self.bias, c0, mask_h)
        #if self.training and (self.rnn_dropout>0):
        #    mask_h = self.get_dropout_mask_((batch, n_out), self.rnn_dropout)
        #    h = h * mask_h.expand_as(h)

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class FastKNN(nn.Module):
    def __init__(self, n_in, n_out, depth, dropout=0, rnn_dropout=0, use_tanh=0):
        super(FastKNN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.depth = depth
        self.dropout = dropout
        self.drop_o = nn.Dropout(dropout)
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = []
        self.seq = nn.Sequential()

        for i in range(depth):
            l = FastKNNCell(
                n_in = n_in if i==0 else n_out,
                n_out = n_out,
                dropout = dropout if i+1 != depth else 0,
                rnn_dropout = rnn_dropout,
                use_tanh = use_tanh
            )
            self.rnn_lst.append(l)
            self.seq.add_module(str(i), l)

    def forward(self, input, c0=None):
        assert input.dim() == 3     # (len, batch, n_in)
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out
            ).zero_())
            c0 = [ zeros for i in range(self.depth) ]
        else:
            assert c0.dim() == 3    # (depth, batch, n_out)
            c0 = c0.chunk(self.depth, 0)

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        return prevx, torch.stack(lstc)

def test_fast(D, L, B, N):
    a = Variable(torch.FloatTensor(L, B, N).zero_().add(0.5).cuda())
    h = Variable(torch.FloatTensor(D, B, N+1).zero_().add(0.5).cuda())
    cell = FastKNN(N, N+1, D).cuda()
    start = time.time()
    for i in range(10000):
        out = cell(a, h)
        out[0][0,0,0].backward()
    print "test1: {:.6f}".format(
        (time.time()-start)/10000
    )

#    K = 4
#    L = 5
#    M = 20
#    D = 20
#    input_pair = (
#        Variable(torch.randn(L,M,D*K).float().cuda(), requires_grad=True),
#        Variable(torch.randn(L,M,D).float().cuda(), requires_grad=True),
#        Variable(torch.randn(D*2).float().cuda(), requires_grad=True),
#        Variable(torch.randn(M,D).float().cuda(), requires_grad=True)
#    )
#    test_grad = gradcheck(KNN_Compute(1), input_pair, eps=1e-3, atol=1e-3)
#    print test_grad

def test_lstm(D, L, B, N):
    a = Variable(torch.FloatTensor(L, B, N).zero_().add(0.5).cuda())
    h = Variable(torch.FloatTensor(D, B, N).zero_().add(0.5).cuda())
    cell = nn.LSTM(N, N, D, dropout=0.0).cuda()
    start = time.time()
    for i in range(10000):
        out = cell(a, (h,h))
        #out[0][0,0,0].backward()
    print "test3: {:.6f}".format(
        (time.time()-start)/10000
    )


if __name__=="__main__":
    test_fast(2, 128, 32, 200)
    #test_lstm(2, 128, 32, 200)


