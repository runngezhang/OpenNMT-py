import torch
import torch.nn as nn
from onmt.modules.Util import BottleLinear
from onmt.modules import aeq


class GlobalAttention(nn.Module):
    """
    Luong Attention.

    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                      a

    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    Loung Attention (dotprod):
    $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

    Bahdanau Attention (mlp):
    $$c = \sum_{j=1}^{SeqLength}\a_jh_j$$.
    The Alignment-function $$a$$ computes an alignment as:
    $$a_j = softmax(v_a^T \tanh(W_a q + U_a h_j) )$$.

    """
    def __init__(self, dim, coverage=False, attn_type="dotprod"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dotprod", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "dotprod":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            self.linear_out = nn.Linear(dim*2, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=False)
            self.v = BottleLinear(dim, 1, bias=False)

        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        self.mask = None

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def applyMask(self, mask):
        self.mask = mask

    def forward_all(self, input, context):
        """
        input (FloatTensor): batch x targetL x dim
        context (FloatTensor): batch x sourceL x dim
        """
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(dim, dim_)
        aeq(batch, batch_)
        aeq(self.dim, dim)
        assert self.mask is None
        assert self.attn_type == "dotprod"

        # (batch, dim, sourceL)
        context_ = context.transpose(1, 2)
        aeq(batch, context_.size(0))
        aeq(dim, context_.size(1))
        aeq(sourceL, context_.size(2))

        if self.attn_type == "dotprod":
            # (batch*targetL, dim)
            input_ = input.view(batch*targetL, dim)
            # (batch, targetL, dim)
            targetT = self.linear_in(input_).view(batch, targetL, dim)
            # (batch, targetL, sourceL)
            attn = torch.bmm(targetT, context_)

        aeq(batch, attn.size(0))
        aeq(targetL, attn.size(1))
        aeq(sourceL, attn.size(2))

        # (batch*targetL, sourceL)
        attn2 = self.sm(attn.view(batch*targetL, sourceL))
        # (batch, targetL, sourceL)
        attn3 = attn2.view(batch, targetL, sourceL)

        # (batch, targetL, dim)
        weightedContext = torch.bmm(attn3, context)

        # Concatenate the input to context (Luong only)
        if self.attn_type == "dotprod":
            weightedContext = torch.cat((weightedContext, input), 2)
            weightedContext = self.linear_out(weightedContext.view(batch*targetL, dim*2))
            weightedContext = self.tanh(weightedContext.view(batch, targetL, dim))

        weightedContext = weightedContext.transpose(0, 1)#.contiguous()
        attn = attn3.transpose(0, 1)#.contiguous()

        #  Check output sizes
        targetL_, batch_, dim_ = weightedContext.size()
        aeq(targetL, targetL_)
        aeq(batch, batch_)
        aeq(dim, dim_)
        targetL_, batch_, sourceL_ = attn.size()
        aeq(targetL, targetL_)
        aeq(batch, batch_)
        aeq(sourceL, sourceL_)

        return weightedContext, attn

    def forward(self, input, context, coverage=None):
        """
        input (FloatTensor): batch x dim
        context (FloatTensor): batch x sourceL x dim
        coverage (FloatTensor): batch x sourceL
        """
        # Check input sizes
        batch, sourceL, dim = context.size()
        batch_, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if self.mask is not None:
            beam_, batch_, sourceL_ = self.mask.size()
            aeq(batch, batch_*beam_)
            aeq(sourceL, sourceL_)

        if coverage:
            context += self.linear_cover(coverage.view(-1).unsqueeze(1)) \
                           .view_as(context)
            context = self.tanh(context)

        # Alignment/Attention Function
        if self.attn_type == "dotprod":
            # batch x dim x 1
            targetT = self.linear_in(input).unsqueeze(2)
            # batch x sourceL
            attn = torch.bmm(context, targetT).squeeze(2)
        elif self.attn_type == "mlp":
            # batch x dim x 1
            wq = self.linear_query(input).unsqueeze(1)
            # batch x sourceL x dim
            uh = self.linear_context(context.contiguous())
            # batch x sourceL x dim
            wquh = uh + wq.expand_as(uh)
            # batch x sourceL x dim
            wquh = self.tanh(wquh)
            # batch x sourceL
            attn = self.v(wquh.contiguous()).squeeze()

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = self.sm(attn)

        # Compute context weighted by attention.
        # batch x 1 x sourceL
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        # batch x dim
        weightedContext = torch.bmm(attn3, context).squeeze(1)

        # Concatenate the input to context (Luong only)
        if self.attn_type == "dotprod":
            weightedContext = torch.cat((weightedContext, input), 1)
            weightedContext = self.linear_out(weightedContext)
            weightedContext = self.tanh(weightedContext)

        # Check output sizes
        batch_, sourceL_ = attn.size()
        aeq(batch, batch_)
        aeq(sourceL, sourceL_)
        batch_, dim_ = weightedContext.size()
        aeq(batch, batch_)
        aeq(dim, dim_)

        return weightedContext, attn
