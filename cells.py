import torch
import torch.nn as nn


class VanillaLSTMCell(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.hid, self.out_dim = hid, hid
        self.x2g = nn.Linear(in_dim, 4 * hid)
        self.h2g = nn.Linear(hid, 4 * hid, bias=False)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, -1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        return o * torch.tanh(c), c


class CIFGCell(nn.Module):  # i = 1 - f
    def __init__(self, in_dim, hid):
        super().__init__()
        self.hid, self.out_dim = hid, hid
        self.x2g = nn.Linear(in_dim, 3 * hid)
        self.h2g = nn.Linear(hid, 3 * hid, bias=False)

    def forward(self, x, state):
        h, c = state
        f, o, g = (self.x2g(x) + self.h2g(h)).chunk(3, -1)
        f, o, g = torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + (1 - f) * g
        return o * torch.tanh(c), c


class LSTMPCell(nn.Module):  # h_t = W_p m_t + b_p, de taille P < H
    def __init__(self, in_dim, hid, proj=64):
        super().__init__()
        self.hid, self.out_dim = hid, proj
        self.x2g = nn.Linear(in_dim, 4 * hid)
        self.h2g = nn.Linear(proj, 4 * hid, bias=False)  # h est de taille P
        self.proj = nn.Linear(hid, proj)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, -1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        m = o * torch.tanh(c)
        return self.proj(m), c


class WMCCell(nn.Module):  # connexions c -> gates, via tanh(W c)
    def __init__(self, in_dim, hid):
        super().__init__()
        self.hid, self.out_dim = hid, hid
        self.x2g = nn.Linear(in_dim, 4 * hid)
        self.h2g = nn.Linear(hid, 4 * hid, bias=False)
        self.ci = nn.Linear(hid, hid, bias=False)
        self.cf = nn.Linear(hid, hid, bias=False)
        self.co = nn.Linear(hid, hid, bias=False)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, -1)
        i = torch.sigmoid(i + torch.tanh(self.ci(c)))
        f = torch.sigmoid(f + torch.tanh(self.cf(c)))
        g = torch.tanh(g)
        c = f * c + i * g
        o = torch.sigmoid(o + torch.tanh(self.co(c)))  # utilise c_t (nouveau)
        return o * torch.tanh(c), c


CELLS = {"vanilla": VanillaLSTMCell, "cifg": CIFGCell,
         "lstmp": LSTMPCell, "wmc": WMCCell}


class Classifier(nn.Module):
    def __init__(self, cell, vocab, emb=128, n_cls=3, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.cell = cell
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cell.out_dim, n_cls)

    def forward(self, x):
        lens = (x != 0).sum(1).clamp(min=1)
        x = x[:, :lens.max()]  # on coupe le padding inutile (plus rapide)
        B, T = x.shape
        e = self.emb(x)
        h = torch.zeros(B, self.cell.out_dim, device=x.device)
        c = torch.zeros(B, self.cell.hid, device=x.device)
        outs = []
        for t in range(T):
            h, c = self.cell(e[:, t], (h, c))
            outs.append(h)
        outs = torch.stack(outs, 1)
        idx = torch.arange(B, device=x.device)
        last = outs[idx, lens - 1]  # dernier token non-pad
        return self.fc(self.drop(last))


def build_model(name, vocab, emb=128, hid=128, proj=64):
    kw = {"proj": proj} if name == "lstmp" else {}
    return Classifier(CELLS[name](emb, hid, **kw), vocab, emb)