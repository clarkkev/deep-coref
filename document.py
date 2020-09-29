import utils
import directories
import timer
from collections import defaultdict
import numpy as np
import evaluation
from collections import Counter


class Document:
    def __init__(self, did, mentions, gold, mention_to_gold):
        self.did = did
        self.mentions = mentions
        self.gold = gold
        self.mention_to_gold = {m: tuple(g) for m, g in mention_to_gold.items()}
        self.reset()

    def reset(self):
        self.clusters = []
        self.mention_to_cluster = {}
        self.rs = {}
        self.ps = {}
        self.ana_to_ant = {}
        self.ant_to_anas = {}
        for m in self.mentions:
            c = (m,)
            self.mention_to_cluster[m] = c
            self.clusters.append(c)
            self.rs[m] = 0
            self.ps[m] = 0
            self.ana_to_ant[m] = -1
            self.ant_to_anas[m] = []
        self.p_num = self.r_num = self.p_den = 0
        self.r_den = sum(len(g) for g in self.gold)

    def get_f1(self, beta=1):
        return evaluation.f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=beta)

    def update_b3(self, c, hypothetical=False):
        timer.start("update b3")
        if len(c) == 1:
            self.p_den -= 1
            self.p_num -= self.ps[c[0]]
            self.r_num -= self.rs[c[0]]
            self.ps[c[0]] = 0
            self.rs[c[0]] = 0
        else:
            intersect_counts = Counter()
            for m in c:
                if m in self.mention_to_gold:
                    intersect_counts[self.mention_to_gold[m]] += 1
            for m in c:
                if m in self.mention_to_gold:
                    self.p_num -= self.ps[m]
                    self.r_num -= self.rs[m]

                    g = self.mention_to_gold[m]
                    ic = intersect_counts[g]
                    self.p_num += ic / float(len(c))
                    self.r_num += ic / float(len(g))
                    if not hypothetical:
                        self.ps[m] = ic / float(len(c))
                        self.rs[m] = ic / float(len(g))
        timer.stop("update b3")

    def link(self, m1, m2, hypothetical=False, beta=1):
        timer.start("link")
        if m1 == -1:
            return self.get_f1(beta=beta) if hypothetical else None

        c1, c2 = self.mention_to_cluster[m1], self.mention_to_cluster[m2]
        assert c1 != c2
        new_c = c1 + c2
        p_num, r_num, p_den, r_den = self.p_num, self.r_num, self.p_den, self.r_den

        if len(c1) == 1:
            self.p_den += 1
        if len(c2) == 1:
            self.p_den += 1
        self.update_b3(new_c, hypothetical=hypothetical)

        if hypothetical:
            f1 = evaluation.f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=beta)
            self.p_num, self.r_num, self.p_den, self.r_den = p_num, r_num, p_den, r_den
            timer.stop("link")
            return f1
        else:
            self.ana_to_ant[m2] = m1
            self.ant_to_anas[m1].append(m2)
            self.clusters.remove(c1)
            self.clusters.remove(c2)
            self.clusters.append(new_c)
            for m in new_c:
                self.mention_to_cluster[m] = new_c
        timer.stop("link")

    def unlink(self, m):
        timer.start("unlink")
        old_ant = self.ana_to_ant[m]
        if old_ant != -1:
            self.ana_to_ant[m] = -1
            self.ant_to_anas[old_ant].remove(m)

            old_c = self.mention_to_cluster[m]
            c1 = [m]
            frontier = self.ant_to_anas[m][:]
            while len(frontier) > 0:
                m = frontier.pop()
                c1.append(m)
                frontier += self.ant_to_anas[m]
            c1 = tuple(c1)
            c2 = tuple(m for m in old_c if m not in c1)

            self.update_b3(c1)
            self.update_b3(c2)

            self.clusters.remove(old_c)
            self.clusters.append(c1)
            self.clusters.append(c2)
            for m in c1:
                self.mention_to_cluster[m] = c1
            for m in c2:
                self.mention_to_cluster[m] = c2
        timer.stop("unlink")


def load_gold(dataset_name):
    gold = {}
    mention_to_gold = {}
    for doc_gold in utils.load_json_lines(directories.GOLD + dataset_name):
        did = int(list(doc_gold.keys())[0])
        gold[did] = doc_gold[str(did)]
        mention_to_gold[did] = {}
        for gold_cluster in doc_gold[str(did)]:
            for m in gold_cluster:
                mention_to_gold[did][m] = tuple(gold_cluster)
    return gold, mention_to_gold


def load_mentions(dataset_name):
    mentions = defaultdict(list)
    mention_ids = np.load(directories.MENTION_DATA + dataset_name + '/mid.npy')
    doc_ids = np.load(directories.MENTION_DATA + dataset_name + '/mdid.npy')
    for did, mid in zip(doc_ids[:, 0], mention_ids[:, 0]):
        mentions[did].append(mid)
    return mentions


def write_docs(dataset_name):
    gold, mention_to_gold = load_gold(dataset_name)
    mentions = load_mentions(dataset_name)
    docs = []
    for did in gold:
        docs.append(Document(did, mentions[did],
                             gold[did], mention_to_gold[did]))
    utils.write_pickle(docs, directories.DOCUMENTS + dataset_name + '_docs.pkl')


def main():
    write_docs("train")
    write_docs("dev")
    write_docs("test")

if __name__ == '__main__':
    main()
