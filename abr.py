from abc import ABCMeta, abstractmethod

# abstract ABR algorithms class
class ABR:

    __metaclass__ = ABCMeta

    @classmethod
    def __init__(self, video, player, save_file):
        self.video = video
        self.player = player
        self.save_file = save_file

    @abstractmethod
    def get_quality(self, network_state): raise NotImplementedError

    @abstractmethod
    def update(self, reward): raise NotImplementedError

