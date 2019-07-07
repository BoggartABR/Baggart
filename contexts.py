from constants import *
from abc import ABCMeta, abstractmethod
import numpy as np


def context_factory(video, player, name):
    if name == BOGGART_CONTEXT:
        return BoggartContext(video, player, name)


class Context:
    __metaclass__ = ABCMeta

    def __init__(self, video, player, name):
        self.video = video
        self.player = player
        self.name = name
        self.predictions = self.video[BITRATE_LEVELS]

    @abstractmethod
    def get_context(self, network_state): raise NotImplementedError

    @abstractmethod
    def get_dimension(self): raise NotImplementedError


class BoggartContext(Context):

    def __init__(self, video, player, name):
        super(BoggartContext, self).__init__(video, player, name)
        self.predictions = np.array([-(self.video[BR_DIM] - 1), -np.floor(np.sqrt(self.video[BR_DIM] - 1)), -1, 0, 1,
                                     np.floor(np.sqrt(self.video[BR_DIM] - 1)), (self.video[BR_DIM] - 1)]).astype(int)

    def get_context(self, network_state):
        batch = network_state[-1]
        next_sizes = batch[NEXT_CHUNKS_START_IDX:NEXT_CHUNKS_END_IDX+1]
        throughput = batch[THROUGHPUT_IDX]
        buffer_size = batch[BUFFER_IDX]

        sizes_in_kb = (np.array(next_sizes) / M_IN_K)
        last_quality = np.argwhere(self.video[BITRATE_LEVELS] == batch[LAST_BR_IDX] * max(self.video[BITRATE_LEVELS]))[0][0]
        qualities_idx = np.array(np.minimum(np.maximum(last_quality + self.predictions, 0), self.video[BR_DIM] - 1))
        qualities = (sizes_in_kb / self.video[VIDEO_CHUNK_LENGTH])[qualities_idx]

        tmp = (np.zeros(len(qualities)) + throughput) - qualities
        tmp = tmp[tmp > 0]
        throughput_idx = max(len(tmp) - 1, 0)
        download_time = np.array(sizes_in_kb / (throughput * M_IN_K))[qualities_idx]
        tmp = (np.zeros(len(qualities_idx)) + buffer_size) - download_time
        tmp = tmp[tmp > 0]
        buffer_idx = max(len(tmp) - 1, 0)

        return throughput_idx, buffer_idx

    def get_dimension(self):
        return len(self.predictions), len(self.predictions)

    def get_predictions(self):
        return self.predictions

    def get_quality(self, last_quality, prediction):
        return max(min(last_quality + self.predictions[prediction], self.video[BITRATE_LEVELS] - 1), 0)
