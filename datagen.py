import numpy as np
import torch
import json
from torch.utils.data import Dataset, random_split
import numpy.linalg as LA


class RoundsDataset(Dataset):
    def __init__(self, roundsjsonlpath: str, fundindustiesjson: str, fundindustryembstsv: str, startupsjsonl: str,
                 startup_industry_embs_tsv: str, funds_jsonl: str):
        with(open(roundsjsonlpath, "r")) as f:
            lines = f.readlines()
            self.rounds = list(map(lambda line: json.loads(line), lines))
        money = np.array(list(map(lambda x: int(x['moneyRaised']), self.rounds)))
        self.fstages = list(set(map(lambda x: x.get('fundingStage', ''), self.rounds)))
        self.ftypes = list(set(map(lambda x: x['fundingType'], self.rounds)))
        self.money_norm = LA.norm(money, 2)

        with(open(funds_jsonl)) as f:
            self.funds = {f['name']: f for f in map(lambda line: json.loads(line), f.readlines())}

        for (name, fund) in self.funds.items():
            if fund.get('country', '') == 'New York':
                fund['country'] = 'United States'

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

        for (name, startup) in self.startups.items():
            if startup.get('country', '') == 'New York':
                startup['country'] = 'United States'

        with(open(startup_industry_embs_tsv)) as f:
            lines = f.readlines()
            self.sie = {}
            for line in lines:
                tokens = line.strip().split("\t")
                industry = tokens[0].replace('#', '')
                vector = np.array(list(map(lambda x: float(x), tokens[1:])))
                self.sie[industry] = vector

        scountries = [startup.get('country', '') for (name, startup) in self.startups.items()]
        fcountries = [fund.get('country', '') for (name, fund) in self.funds.items()]
        self.countries = sorted(list(filter(lambda x: x != '', set(scountries + fcountries))))

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

        vecs = np.array([self.fie[industry] * weight for (industry, weight) in industries.items()])
        vs = np.sum(vecs, axis=0)
        n = LA.norm(vs, 2)
        vs /= n
        return vs

    def _investors_emb(self, funds) -> np.ndarray:
        for fund in funds:
            assert fund in self.fund2industries
        assert funds is not None
        fv = np.array([self._fund_vector(fund) for fund in funds])
        investors_emb = np.sum(fv, axis=0)
        n = LA.norm(investors_emb, 2)
        investors_emb /= n
        return investors_emb

    def __getitem__(self, item: int):
        r = self.rounds[item]
        startup_emb = self._startup_vector(r['company'])
        investors_emb = self._investors_emb(r['investors'])
        money = int(r['moneyRaised']) / self.money_norm
        ts = torch.tensor(startup_emb.astype(np.float32))
        ti = torch.tensor(investors_emb.astype(np.float32))

        fstage = self.fstages.index(r.get('fundingStage', ''))
        ftype = self.ftypes.index(r['fundingType'])
        tfstage = torch.zeros(len(self.fstages))
        tftype = torch.zeros(len(self.ftypes))
        tfstage[fstage] = 1
        tftype[ftype] = 1

        sc = self.startups[r['company']].get('country', '')
        fcs = set(map(lambda f: self.funds[f].get('country', ''), filter(lambda f: f in self.funds, r['investors'])))
        tsc = torch.zeros(len(self.countries))
        tfcs = torch.zeros(len(self.countries))
        if sc in self.countries:
            tsc[self.countries.index(sc)] = 1
        for fc in fcs:
            if fc in self.countries:
                tfcs[self.countries.index(fc)] = 1

        year = int(r.get('dealDate', r.get('announcedDate', '0')).split('-')[0])
        # ty = torch.tensor([year])

        tm = torch.tensor(money)
        # return torch.cat([ts, ti, tfstage, tftype, tsc, tfcs], dim=0), tm
        return torch.cat([ts, ti, tfstage, tftype], dim=0), tm


if __name__ == '__main__':
    rd = RoundsDataset('dataset/supercleanrounds.jsonl',
                       'dataset/fund_industries.json',
                       'dataset/fund_industries.tsv',
                       'dataset/startups.jsonl',
                       'dataset/startup_industries.tsv',
                       'dataset/funds.jsonl')
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
