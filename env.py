import numpy as np
from random import Random
from constants import *


# BITRATE_LEVELS = 9 #########
# TOTAL_VIDEO_CHUNCK = 116 ########
# VIDEO_BIT_RATE = [200, 300, 480, 750, 1200, 1850, 2850, 4300, 5300]
# VIDEO_CHUNCK_LEN = 2005.0  # millisec, every time add this amount to buffer
# VIDEO_SIZE_FILE = './creepy_video/video_size_'


DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
NOISE_LOW = 0.9
NOISE_HIGH = 1.1



class Environment:
    def __init__(self, traces, type, video, random_seed=None):

        self.myRandom = Random(random_seed)

        self.all_cooked_time = traces[TIME]
        self.all_cooked_bw = traces[BANDWIDTH]

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.type = type

        self.reset_video()

        self.video = video


    def reset_video(self):
        # pick a trace file and start point according to environment type
        if self.type == TRAIN:
            self.trace_idx = 0
            self.mahimahi_ptr = 1
        elif self.type == TEST:
            self.trace_idx = self.myRandom.randint(0, len(self.all_cooked_time) - 1)
            self.mahimahi_ptr = self.myRandom.randint(1, len(self.cooked_bw) - 1)
        else:
            raise Exception("unknow environment type")

        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]


    def get_bitrate_size_ratio(self):
        return self.video[BITRATE_SIZE_RATIO]

    def get_video_chunk_sizes(self):
        values = np.array(self.video[VIDEO_SIZES].values())
        return values[:, self.video[NUM_OF_CHUNKS]]


    # simulates video download without updating network parameters.
    # returns the time it took to download the video
    def simulate_download(self, quality):

        video_chunk_size = self.video[VIDEO_SIZES][quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        start_ptr = self.mahimahi_ptr
        last_time = self.last_mahimahi_time

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[start_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[start_ptr] - last_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            last_time = self.cooked_time[start_ptr]
            start_ptr += 1

            if start_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0

                start_ptr = 1
                last_time = 0

        buffer = self.buffer_size

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= self.myRandom.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - buffer, 0.0)

        # update the buffer
        buffer = np.maximum(buffer - delay, 0.0)

        # add in the new chunk
        buffer += self.video[VIDEO_CHUNK_LENGTH]

        # sleep if buffer gets too large
        sleep_time = 0
        if buffer > BUFFER_CAPACITY * MILLISECONDS_IN_SECOND:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = buffer - BUFFER_CAPACITY * MILLISECONDS_IN_SECOND
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
            buffer -= sleep_time

            while True:
                duration = self.cooked_time[start_ptr] - last_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    last_time += sleep_time / MILLISECONDS_IN_SECOND
                    break

                sleep_time -= duration * MILLISECONDS_IN_SECOND
                last_time = self.cooked_time[start_ptr]
                start_ptr += 1

                if start_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0

                    start_ptr = 1
                    last_time = 0


        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video

        video_chunk_remain = self.video[VIDEO_CHUNK_LENGTH] - self.video_chunk_counter - 1

        end_of_video = False
        if self.video_chunk_counter + 1 >= self.video[NUM_OF_CHUNKS]:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            self.reset_video()

        next_video_chunk_sizes = []
        for i in xrange(BITRATE_LEVELS):
            next_video_chunk_sizes.append(np.array(self.video_size[i][self.video_chunk_counter + 1]))

        last_bitrate = self.video[BITRATE_LEVELS][quality] / float(np.max(self.video[BITRATE_LEVELS]))
        throughput = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        norm_delay = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        remaining_chunks = np.minimum(video_chunk_remain, self.video[NUM_OF_CHUNKS]) / float(self.video[NUM_OF_CHUNKS])
        next_sizes = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K

        batch = np.zeros(INPUT_LEN)
        batch[LAST_BR_IDX] = last_bitrate
        batch[BUFFER_IDX] = buffer / MILLISECONDS_IN_SECOND
        batch[THROUGHPUT_IDX] = throughput
        batch[DELAY_IDX] = norm_delay
        batch[NEXT_CHUNKS_START_IDX:NEXT_CHUNKS_END_IDX+1] = next_sizes
        batch[CHUNKS_TILL_END_IDX] = remaining_chunks

        return batch, sleep_time, rebuf / MILLISECONDS_IN_SECOND, end_of_video, start_ptr, last_time


    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.video[BR_DIM]

        batch, sleep_time, rebuf, end_of_video, ptr, time = self.simulate_download(quality)

        self.video_chunk_counter += 1
        self.last_mahimahi_time = time
        self.mahimahi_ptr = ptr
        self.buffer_size = batch[1] * MILLISECONDS_IN_SECOND

        return batch, sleep_time, rebuf, end_of_video
