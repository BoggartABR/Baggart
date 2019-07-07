from constants import *
from abc import ABCMeta, abstractmethod
import numpy as np


def get_reward_class(video, type, rebuf_penalty  = None, smooth_penalty = SMOOTH_PENALTY):

    if type == LIN:
        return LinReward(video, type, rebuf_penalty, smooth_penalty)
    if type == HD:
        return HD_REWARD(video, type, rebuf_penalty, smooth_penalty)
    if type == LOG:
        return LogReward(video, type, rebuf_penalty, smooth_penalty)


class Reward():
    __metaclass__ = ABCMeta

    @classmethod
    def init_session(self, video, type, rebuf_penalty  = None, smooth_penalty = SMOOTH_PENALTY):
        self.type = type
        self.video = video
        if rebuf_penalty:
            self.rebuf_penalty = rebuf_penalty
        else:
            self.rebuf_penalty = max(video[BITRATE_LEVELS] / M_IN_K)

        self.smooth_penalty = smooth_penalty

    @abstractmethod
    def mapping(self, bitrate): raise NotImplementedError

    def norm_reward(self, quality, last_quality, rebuf):
        min_reward = self.mapping(min(self.video[BITRATE_LEVELS])) \
                     - self.rebuf_penalty * MAX_REBUF \
                     - self.smooth_penalty \
                     * (self.mapping(max(self.video[BITRATE_LEVELS])) - self.mapping(min(self.video[BITRATE_LEVELS])))

        max_reward = self.mapping(max(self.video[BITRATE_LEVELS]))
        return (self.reward(quality, last_quality, rebuf) - min_reward) / (max_reward - min_reward)

    def reward(self, quality, last_quality, rebuf):
        reward = self.mapping(self.video[BITRATE_LEVELS][quality]) \
                 - self.rebuf_penalty * min(rebuf, MAX_REBUF) \
                 - self.smooth_penalty * np.abs(self.mapping(self.video[BITRATE_LEVELS][quality])
                                                - self.mapping(self.video[BITRATE_LEVELS][last_quality]))
        return reward


class LinReward(Reward):

    def mapping(self, bitrate):
        return bitrate / M_IN_K

class HDReward(Reward):

    def mapping(self, bitrate):
        return self.video[HD_REWARD][np.argwhere(self.video[BITRATE_LEVELS] == bitrate)[0][0]]

class LogReward(Reward):

    def mapping(self, bitrate):
        return np.log(bitrate / float(self.video[BITRATE_LEVELS][0]))

