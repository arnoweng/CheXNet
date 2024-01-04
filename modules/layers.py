import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide']

def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output

def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output

class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha = 1):
        return R
    def m_relprop(self, R,pred,  alpha = 1):
        return R
    def RAP_relprop(self, R_p):
        return R_p

# An ordinary implementation of Swish function
class Swish(RelProp):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(RelProp):
    def forward(self, x):
        return SwishImplementation.apply(x)


class RelPropSimple(RelProp):
    def relprop(self, R, alpha = 1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)[0]
            if torch.is_tensor(self.X) == False:
                Rp = []
                Rp.append(self.X[0] * Cp)
                Rp.append(self.X[1] * Cp)
            else:
                Rp = self.X * (Cp)
            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Identity(nn.Identity, RelProp):
    pass

class ReLU(nn.ReLU, RelProp):
    pass

class LeakyReLU(nn.LeakyReLU, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass

class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1):
        px = torch.clamp(self.X, min=0)

        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)

            S1 = safe_divide(R, Z1)

            C1 = x1 * self.gradprop(Z1, x1, S1)[0]

            return C1

        activator_relevances = f(px)


        out = activator_relevances

        return out

class ZeroPad2d(nn.ZeroPad2d, RelPropSimple):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        outputs = self.X * C[0]
        return outputs

class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

'''여기는 생각해보자'''
class Multiply(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)

    def relprop(self, R, alpha = 1):
        x0 = torch.clamp(self.X[0],min=0)
        x1 = torch.clamp(self.X[1],min=0)
        x = [x0,x1]
        Z = self.forward(x)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, x, S)

        outputs = []
        outputs.append(x[0] * C[0])
        outputs.append(x[1] * C[1])

        return outputs

class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha = 1):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = []
            for _ in range(self.num):
                Z.append(self.X)

            Spp = []
            Spn = []

            for z, rp, rn in zip(Z, R_p):
                Spp.append(safe_divide(torch.clamp(rp, min=0), z))
                Spn.append(safe_divide(torch.clamp(rp, max=0), z))

            Cpp = self.gradprop(Z, self.X, Spp)[0]
            Cpn = self.gradprop(Z, self.X, Spn)[0]

            Rp = self.X * (Cpp * Cpn)

            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha = 1):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs
    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X, self.dim)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)

            Rp = []

            for x, cp in zip(self.X, Cp):
                Rp.append(x * (cp))


            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class Sequential(nn.Sequential):
    def relprop(self, R, alpha = 1):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R
    def RAP_relprop(self, Rp):
        for m in reversed(self._modules.values()):
            Rp = m.RAP_relprop(Rp)
        return Rp

class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha = 1):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R
    def RAP_relprop(self, R_p):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1
        def backward(R_p):
            X = self.X

            weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))

            if torch.is_tensor(self.bias):
                bias = self.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
                                     R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if torch.is_tensor(self.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha = 1):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        # def f(w1, w2, x1, x2):
        #     Z1 = F.linear(x1, w1)
        #     Z2 = F.linear(x2, w2)
        #     S1 = safe_divide(R, Z1)
        #     S2 = safe_divide(R, Z2)
        #     C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        #     C2 = x2 * self.gradprop(Z2, x2, S2)[0]
        #     return C1 #+ C2

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            Z = Z1 + Z2
            S = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S)[0]
            C2 = x2 * self.gradprop(Z2, x2, S)[0]
            return C1 + C2


        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        out = alpha * activator_relevances - beta*inhibitor_relevances

        return out

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n

            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=-1,keepdim=True)-R.sum(dim=-1,keepdim=True))
            return C
        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.linear(x1, w1) * R_nonzero
            Za2 = - F.linear(x1, w2) * R_nonzero

            Zb1 = - F.linear(x2, w1) * R_nonzero
            Zb2 = F.linear(x2, w2) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)

            return C1 + C2

        def first_prop(pd, px, nx, pw, nw):
            Rpp = F.linear(px, pw) * pd
            Rpn = F.linear(px, nw) * pd
            Rnp = F.linear(nx, pw) * pd
            Rnn = F.linear(nx, nw) * pd
            Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
            Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

            Z1 = F.linear(px, pw)
            Z2 = F.linear(px, nw)
            Z3 = F.linear(nx, pw)
            Z4 = F.linear(nx, nw)

            S1 = safe_divide(Rpp, Z1)
            S2 = safe_divide(Rpn, Z2)
            S3 = safe_divide(Rnp, Z3)
            S4 = safe_divide(Rnn, Z4)
            C1 = px * self.gradprop(Z1, px, S1)[0]
            C2 = px * self.gradprop(Z2, px, S2)[0]
            C3 = nx * self.gradprop(Z3, nx, S3)[0]
            C4 = nx * self.gradprop(Z4, nx, S4)[0]
            bp = self.bias * pd * safe_divide(Pos, Pos + Neg)
            bn = self.bias * pd * safe_divide(Neg, Pos + Neg)
            Sb1 = safe_divide(bp, Z1)
            Sb2 = safe_divide(bn, Z2)
            Cb1 = px * self.gradprop(Z1, px, Sb1)[0]
            Cb2 = px * self.gradprop(Z2, px, Sb2)[0]
            return C1 + C4 + Cb1 + C2 + C3 + Cb2
        def backward(R_p, px, nx, pw, nw):
            # dealing bias
            # if torch.is_tensor(self.bias):
            #     bias_p = self.bias * R_p.ne(0).type(self.bias.type())
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp
        def redistribute(Rp_tmp):
            Rp = torch.clamp(Rp_tmp, min=0)
            Rn = torch.clamp(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        X = self.X
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  ## first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A =  redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)


        return Rp


class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha = 1):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)


            # def f(w1, w2, x1, x2):
            #     Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            #     Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            #     S1 = safe_divide(R, Z1)
            #     S2 = safe_divide(R, Z2)
            #     C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            #     C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            #      return C1 + C2

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2



            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1,2,3], keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=[1,2,3], keepdim=True) - R.sum(dim=[1,2,3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero
            Za2 = - F.conv2d(x1, w2, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero

            Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero
            Zb2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)
            return C1 #+ C2

        def backward(R_p, px, nx, pw, nw):

            # if torch.is_tensor(self.bias):
            #     bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            #     bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
            #                          R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp
        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        return Rp

def get_same_padding_conv2d(image_size):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    return partial(Conv2dStaticSamePadding, image_size=image_size)

# def get_same_padding_depthwise_conv2d(image_size):
#     """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
#        Static padding is necessary for ONNX exporting of models.
#     Args:
#         image_size (int or tuple): Size of the image.
#     Returns:
#         Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
#     """
#     return partial(Conv2_depthwise_dStaticSamePadding, image_size=image_size)

class ConvTranspose2d(nn.ConvTranspose2d, RelProp):
    def relprop(self, R, alpha = 1):

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, x1):
            Z1 = F.conv_transpose2d(x1, w1, bias=None, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            return C1

        activator_relevances = f(pw, px)
        R = activator_relevances
        return R

class Conv2dStaticSamePadding(Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.pad_flag = 'zeropad'
            self.static_padding = ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.pad_flag = 'identity'
            self.static_padding = Identity()

    def forward(self, x):
        self.X = x
        self.padd_output = self.static_padding(self.X)
        x = F.conv2d(self.padd_output, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def relprop(self, R, alpha=1):
        if self.padd_output.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.padd_output
            L = self.padd_output * 0 + \
                torch.min(torch.min(torch.min(self.padd_output, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.padd_output * 0 + \
                torch.max(torch.max(torch.max(self.padd_output, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.padd_output, min=0)
            nx = torch.clamp(self.padd_output, max=0)

            # def f(w1, w2, x1, x2):
            #     Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            #     Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            #     S1 = safe_divide(R, Z1)
            #     S2 = safe_divide(R, Z2)
            #     C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            #     C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            #      return C1 + C2

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        R = self.static_padding.relprop(R)
        return R

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1, 2, 3], keepdim=True)) * torch.ne(R, 0).type(
                R.type())
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=[1, 2, 3], keepdim=True) - R.sum(dim=[1, 2, 3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero
            Za2 = - F.conv2d(x1, w2, bias=None, stride=self.stride, padding=self.padding,
                             groups=self.groups) * R_nonzero

            Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.stride, padding=self.padding,
                             groups=self.groups) * R_nonzero
            Zb2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)
            return C1  # + C2

        def backward(R_p, px, nx, pw, nw):

            # if torch.is_tensor(self.bias):
            #     bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            #     bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
            #                          R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp

        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.padd_output, min=0)
        nx = torch.clamp(self.padd_output, max=0)

        if self.padd_output.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.padd_output)
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        Rp = self.static_padding.relprop(Rp)
        return Rp




if __name__=='__main__':

    convt = ConvTranspose2d(100, 50, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False).cuda()

    rand = torch.rand((1,100,224,224)).cuda()
    out = convt(rand)
    rel = convt.relprop(out)

    print(out.shape)