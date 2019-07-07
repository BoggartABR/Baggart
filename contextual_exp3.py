import exp3
import pickle


def calc_inner_index(dim, idx):
    assert len(dim) == len(idx)
    if len(dim) == 0:
        return 0
    tmp = 1
    for i in dim[1:]:
        tmp *= i
    return idx[0] * tmp + calc_inner_index(dim[1:], idx[1:])


class Contextual_Exp3:

    def __init__(self, context_dim, num_of_arms, to_save = None):
        self.num_of_arms = num_of_arms
        self.context_dim = context_dim
        self.flat_dim = 1
        for i in context_dim:
            self.flat_dim *= i
        self.exp3_list = [None] * self.flat_dim
        self.first_round = False
        self.last_exp3 = None
        self.last_context = None
        self.to_save = to_save


    def predict(self, context):

        flat_idx = calc_inner_index(self.context_dim, context)
        self.contexts_counter[flat_idx] += 1

        if self.exp3_list[flat_idx] is None:
            self.exp3_list[flat_idx] = exp3.Exp3(self.num_of_arms, context, self.save_contexts)
            self.first_round = True
        else:
            self.first_round = False
        self.last_exp3 = self.exp3_list[flat_idx]
        self.last_context = context
        arm = self.exp3_list[flat_idx].predict()
        return arm

    def update(self, reward):
        self.last_exp3.update(reward)


    def save(self):
        if self.to_save is None:
            return
        with open(self.to_save, 'wb') as f:
            pickle.dump(self, f)
