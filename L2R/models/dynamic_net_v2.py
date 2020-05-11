from enum import Enum
import torch
#import pickle
import torch.nn as nn

class ForwardType(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3

class DynamicNet_v2(object):
    def __init__(self, c0, lr):
        self.models = []
        self.boost_rates = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate  = nn.Parameter(torch.tensor(lr, requires_grad=True, device="cuda"))

    def add(self, model, boost_rate):
        self.models.append(model)
        self.boost_rates.append(boost_rate)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())
        
        for br in self.boost_rates:
            params.append(br)

        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

        for br in self.boost_rates:
            br.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        prediction = None
        cnt=0
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(x, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(x, middle_feat_cum)
                    prediction += self.boost_rates[cnt]*pred
                cnt += 1
        return middle_feat_cum, self.c0 + prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        prediction = None
        cnt=0
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += self.boost_rates[cnt]*pred
            cnt += 1
        return middle_feat_cum, self.c0 + prediction

    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet_v2(d['c0'], d['lr'])
        net.boost_rate = d['boost_rate']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod, net.boost_rate)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate}
        torch.save(d, path)
