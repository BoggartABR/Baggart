from constants import *
from abr import ABR
import contextual_exp3
import numpy as np
from contexts import context_factory

class Boggart(ABR):

    def __init__(self, video, player, save_file, context_name = "boggart"):
        super(Boggart, self).__init__(video, player, save_file)
        self.context_generator = context_factory(context_name)
        self.trainer = contextual_exp3.Contextual_Exp3(self.context_generator.get_dim,
                                                       len(self.context_generator.get_predictions), save_file)

    def get_quality(self, network_state):
        context = self.context_generator(network_state)
        prediction = self.trainer.predict(context)
        last_quality = np.argwhere(self.video[BITRATE_LEVELS] == network_state[-1][LAST_BR_IDX] *
                                   max(self.video[BITRATE_LEVELS]))[0][0]
        quality = self.context_generator.get_quality(last_quality, prediction)
        return quality

    def update(self, reward):
        self.trainer.update(reward)

    def save_params(self):
        self.trainer.save()
