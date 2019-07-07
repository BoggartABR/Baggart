from abr import ABR
from constants import *


RESEVOIR = 5
CUSHION = 10

class BBA(ABR):

    def get_quality(self, network_state):
        last_batch = network_state[-1]
        buffer_size = last_batch[BUFFER_IDX]

        if buffer_size < RESEVOIR:
            quality = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            quality = len(self.bitrate_list) - 1
        else:
            quality = (len(self.bitrate_list) - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        return int(quality)

    def update(self, reward):
        return