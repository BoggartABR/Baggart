from constants import *
import abr
import numpy as np
import os
from env import Environment
import rewards
import logging
from constants import *
import sys
import json

LOG_DIR = 'results/'
VIDEO = ''


# initializes parameters of played video
def init_video(video_file):

    with open(video_file) as vid:
        manifest = json.load(vid)

    video = {}
    video[NUM_OF_CHUNKS] = len(manifest["segment_sizes_bits"][0])
    video[BITRATE_LEVELS] = manifest["bitrates_kbps"]
    video[HD_REWARD] = manifest["hd_rewards"]
    video[VIDEO_CHUNK_LENGTH] = manifest["bitrates_kbps"] / M_IN_K

    video[BR_DIM] = len(video[BITRATE_LEVELS])
    video[VIDEO_SIZES] = manifest["segment_sizes_bits"] / M_IN_K / M_IN_K / BITS_IN_BYTE
    return video


def load_trace(trace_folder):

    traces = {}

    cooked_files = os.listdir(trace_folder)
    cooked_files.sort()
    traces[TIME] = []
    traces[BANDWIDTH] = []
    traces[FILE_NAMES] = []
    for cooked_file in cooked_files:
        file_path = trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        traces[TIME].append(cooked_time)
        traces[BANDWIDTH].append(cooked_bw)
        traces[FILE_NAMES].append(cooked_file)

    return traces

def load_ABR(ABR_name):
    return 0

def load_logging(total_log, video_log, logging_interval, num_of_qualities):
    return logging.Log(total_log, video_log, logging_interval, num_of_qualities)


abr = sys.argv[0]
trace_dir = sys.argv[1]
reward_type = sys.argv[2]
log_file = LOG_DIR + abr + trace_dir + reward_type

video = init_video(VIDEO)

for trace in os.listdir(trace_dir):
    traces = load_trace(trace)

    alg = load_ABR(abr)
    env = Environment(traces, type, video, RANDOM_SEED)

    reward = rewards.get_reward_class(video, reward_type)

    quality = DEFAULT_QUALITY
    last_quality = DEFAULT_QUALITY
    batch = np.zeros((HISTORY_LEN, INPUT_LEN))
    video_count = 0
    t = 0

    while True:

        last_batch, sleep_time, rebuf, end_of_video = env.get_video_chunk(quality)
        remaining_chunks = last_batch[CHUNKS_TILL_END_IDX] * video[NUM_OF_CHUNKS]
        norm_reward = reward.norm_reward(quality, last_quality, rebuf)

        qualities = []

        first_round = False
        their_reward = reward.reward(quality, last_quality, rebuf)
        last_quality = quality

        batch = np.roll(batch, -1, axis=0)
        batch[-1] = last_batch

        quality = alg.predict(batch)

        if end_of_video:
            last_quality = DEFAULT_QUALITY
            quality = DEFAULT_QUALITY
            video_count += 1

            if type == TEST and video_count > len(traces[FILE_NAMES]):
                break
        t += 1

