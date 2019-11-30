import numpy as np
import torch
import json
from torch.utils.data import Dataset, random_split
import numpy.linalg as LA


class RoundsDataset(Dataset):
    def __init__(self, roundsjsonlpath: str, fundindustiesjson: str, fundindustryembstsv: str, startupsjsonl: str,
                 startup_industry_embs_tsv: str):
        with(open(roundsjsonlpath, "r")) as f:
            lines = f.readlines()
            self.rounds = list(map(lambda line: json.loads(line), lines))
        money = np.array(list(map(lambda x: int(x['moneyRaised']), self.rounds)))
        self.fstages = list(set(map(lambda x: x.get('fundingStage', ''), self.rounds)))
        self.ftypes = list(set(map(lambda x: x['fundingType'], self.rounds)))
        self.money_norm = LA.norm(money, 2)

        with(open(fundindustiesjson)) as f:
            self.fund2industries = json.load(f)

        with(open(fundindustryembstsv)) as f:
            lines = f.readlines()
            self.fie = {}
            for line in lines:
                tokens = line.strip().split("\t")
                industry = tokens[0].replace('#', '')
                vector = np.array(list(map(lambda x: float(x), tokens[1:])))
                self.fie[industry] = vector

        with(open(startupsjsonl, "r")) as f:
            lines = f.readlines()
            self.startups = {s['company']: s for s in map(lambda line: json.loads(line), lines)}

        with(open(startup_industry_embs_tsv)) as f:
            lines = f.readlines()
            self.sie = {}
            for line in lines:
                tokens = line.strip().split("\t")
                industry = tokens[0].replace('#', '')
                vector = np.array(list(map(lambda x: float(x), tokens[1:])))
                self.sie[industry] = vector

    def __len__(self):
        return len(self.rounds)

    def _startup_vector(self, company):
        s = self.startups[company]
        assert s is not None
        industries = list(map(lambda ind: ind.lower().replace(" ", "_"), s['industries']))
        for industry in industries:
            assert industry in self.sie

        vecs = np.array([self.sie[industry] for industry in industries])
        vs = np.sum(vecs, axis=0)
        n = LA.norm(vs, 2)
        vs /= n
        return vs

    def _fund_vector(self, fund):
        industries = self.fund2industries[fund]
        assert industries is not None
        for industry in industries:
            assert industry in self.fie

        vecs = np.array([self.fie[industry] * industries[industry] for industry in industries])
        vs = np.sum(vecs, axis=0)
        n = LA.norm(vs, 2)
        vs /= n
        return vs

    def __getitem__(self, item: int):
        r = self.rounds[item]
        c = r['company']
        startup_emb = self._startup_vector(c)

        funds = r['investors']
        for fund in funds:
            assert fund in self.fund2industries
        assert funds is not None
        fv = np.array([self._fund_vector(fund) for fund in funds])
        investors_emb = np.sum(fv, axis=0)
        n = LA.norm(investors_emb, 2)
        investors_emb /= n
        money = int(r['moneyRaised']) / self.money_norm
        ts = torch.tensor(startup_emb.astype(np.float32))
        ti = torch.tensor(investors_emb.astype(np.float32))
        tfstage = torch.tensor([self.fstages.index(r.get('fundingStage', '')) / len(self.fstages)])
        tftype = torch.tensor([self.ftypes.index(r['fundingType']) / len(self.ftypes)])
        tm = torch.tensor(money)
        return torch.cat([ts, ti, tfstage, tftype], dim=0), tm


if __name__ == '__main__':
    rd = RoundsDataset('/home/zhukov/clients/unicorn/unicorn/supercleanrounds.jsonl',
                       '/home/zhukov/clients/unicorn/unicorn/fund_industries.json',
                       '/home/zhukov/clients/unicorn/Starspace/fund2inv2.tsv',
                       '/home/zhukov/clients/unicorn/unicorn/startups.jsonl',
                       '/home/zhukov/clients/unicorn/Starspace/industries.tsv')
    x = rd[1]
    print(x)

    train_size = len(rd) * 80 // 100
    validation_size = len(rd) - train_size
    datasets = random_split(rd, [train_size, validation_size])
    print(len(datasets[0]), len(datasets[1]))

    mmin = 1
    mmax = 0
    for (x, money) in rd:
        m = money.item()
        print(m)
        mmin = min(mmin, m)
        mmax = max(mmax, m)

    print(mmin, mmax, mmin * rd.money_norm, mmax * rd.money_norm)
