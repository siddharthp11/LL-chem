import numpy as np

class FGVocab():
    def __init__(self):
        self.fg_to_idx = {}
        self.vocab_size = 0

    def add_to_vocab(self, fg):
        if fg not in self.fg_to_idx.keys(): 
            self.fg_to_idx[fg] = self.vocab_size
            self.vocab_size += 1

    def vectorize_reaction(self, reaction):
        r, p = reaction
        rv, pv = np.zeros(self.vocab_size), self.get_vector(p)
        for reactant in r: rv = rv + self.get_vector(reactant)
        return rv.tolist(), pv.tolist()

    def get_vector(self, reactant):
        fg_vector = np.zeros(self.vocab_size)
        for k in range(len(reactant)):
            fg, count = reactant[k]
            fg_idx = self.fg_to_idx.get(fg)
            if fg_idx: fg_vector[fg_idx] = count
        return fg_vector