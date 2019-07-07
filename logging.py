import numpy as np

class Log:

    def __init__(self, total_log, video_log, logging_interval, num_of_qualities):
        self.total_log = total_log
        self.video_log = video_log
        self.logging_interval = logging_interval
        self.qualities = np.zeros(num_of_qualities)

        self.rewards = []
        self.bitrates = []
        self.rebufs = []
        self.smoothness = []

        self.total_rewards = []
        self.total_bitrates = []
        self.total_rebufs = []
        self.total_smoothness = []

        self.videos_rewards = []
        self.videos_bitrates = []
        self.videos_rebufs = []
        self.video_smoothness = []


    def update(self, t, reward, bitrate, rebuf, smoothness, quality, end_of_video):

        self.rewards.append(reward)
        self.bitrates.appens(bitrate)
        self.rebufs.append(rebuf)
        self.smoothness.append(smoothness)

        self.total_rewards.append(reward)
        self.total_bitrates.appen(bitrate)
        self.total_rebufs.append(rebuf)
        self.total_smoothness.append(smoothness)

        self.qualities[quality] += 1

        if end_of_video:
            self.videos_rewards.append(sum(self.rewards) / float(len(self.rewards)))
            self.videos_bitrates.appens(sum(self.bitrates) / float(len(self.bitrates)))
            self.videos_rebufs.append(sum(self.rebufs) / float(len(self.rebufs)))
            self.video_smoothness.append(sum(self.smoothness) / float(len(self.smoothness)))

            self.rewards = []
            self.bitrates = []
            self.rebufs = []
            self.smoothness = []

        if t % self.logging_interval == 0:
            self.log(t)
            self.total_rewards = []
            self.bitrates = []
            self.total_rebufs = []
            self.total_smoothness = []
            self.qualities = np.zeros(len(self.qualities))

    def log(self, t):
        print "after ", t, "iterations, qualities count: ", self.qualities
        with open(self.total_log, 'a') as f:
            f.write("mean rewards:" + str(sum(self.total_rewards) / float(len(self.total_rewards))) + '\n')
            f.write("mean bitrates:" + str(sum(self.total_bitrates) / float(len(self.total_bitrates))) + '\n')
            f.write("mean rebufs:" + str(sum(self.total_rebufs) / float(len(self.total_rebufs))) + '\n')
            f.write("mean smoothness:" + str(sum(self.total_smoothness) / float(len(self.total_smoothness))) + '\n')

    def log_videos(self):
        with open(self.video_log, 'a') as f:
            f.write('reward:' + '\n')
            for r in self.videos_rewards.tolist():
                f.write(str(r) + ' ')
            f.write('\n' + 'bitrates:' + '\n')
            for q in self.videos_bitrates.tolist():
                f.write(str(q) + ' ')
            f.write('\n' + 'rebufs:' + '\n')
            for r in self.videos_rebufs.tolist():
                f.write(str(r) + ' ')
            f.write('\n' + 'smoothness:' + '\n')
            for s in self.videos_smoothness.tolist():
                f.write(str(s) + ' ')
            f.write('\n')



