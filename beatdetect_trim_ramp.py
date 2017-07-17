import sys, os
import glob
from scikits.audiolab import *
from scipy.io import wavfile
from scipy.signal import bilinear, lfilter
from numpy import pi, polymul
import numpy as np
import madmom
from math import *
from tqdm import tqdm
import argparse

#Jeff Mentch 7/14/17
#script to:
#1)split recorded Pandora streams into seperate tracks (note: AdBlock+ blocks Ads beforehand)
#2)selects three ~6.32 second clip from each track which are
#	a)one from each third of the track
#	b)on a probable downbeat (Bock et. al RNN)
#	c)50ms quarter sin ramp up and ramp down
#3)exports as 44.1khz 16bit PCM stereo .wav
#
#Silence detection and splitting adapted from https://gist.github.com/rudolfbyker/8fc0d99ecadad0204813d97fee2c6c06
#
#Joint Beat and Downbeat Tracking with Recurrent Neural Networks from:
#Sebastian Bock, Florian Krebs and Gerhard Widmer,
#Proceedings of the 17th International Society for Music Information Retrieval Conference (ISMIR), 2016.




def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1
        
def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in xrange(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]



parser = argparse.ArgumentParser(description='Split and process recorded Pandora streams')
parser.add_argument('indir', type=str, default='/Users/jeff/Documents/increase_prior/recordings/original', help='The input directory (with .wav files), default is /Users/jeff/Documents/increase_prior/recordings/original')
parser.add_argument('outdir', type=str, default='/Users/jeff/Documents/increase_prior/recordings/split', help='The output dir. Defaults to /Users/jeff/Documents/increase_prior/recordings/split')
parser.add_argument('min_silence_length',type=float, default=0.5, help='Minimum length of silence for a split. Defaults to 0.5 seconds.')
parser.add_argument('silence_threshold', type=float, default=1e-6, help='energy level (between 0.0 and 1.0) below which the signal is regarded as silent. Defaults to 1e-6 == 0.0001%.')
parser.add_argument('step_duration', type=float, default=None, help='amount of time to step through input file after calculating e. Smaller = slower, more accurate; Larger = faster, might miss. Default is min_silence_length/10')

parser.add_argument('--dry-run', action='store_true', help='Don\'t actually write any output files.')


args = parser.parse_args()
window_duration=args.min_silence_length
if args.step_duration is None:
	step_duration=args.min_silence_length/10 #
else:
    step_duration = args.step_duration


print "Splitting where energy is below {}% for longer than {}s.".format(
    silence_threshold * 100.,
    window_duration
)


file_list = glob.glob("%s/*.wav")%indir
if len(file_list) < 1:
        print "run in a directory with .wav files"
        sys.exit(1)


#load file
sample_rate, samples = wavfile.read(filename="/Users/jeff/Documents/increase_prior/recordings/original/a_001.wav", mmap=True)

max_amplitude = np.iinfo(samples.dtype).max
max_energy = energy([max_amplitude])


window_size = int(window_duration * sample_rate)
step_size = int(step_duration * sample_rate)

signal_windows = windows(
    signal=samples,
    window_size=window_size,
    step_size=step_size
)

window_energy = (energy(w) / max_energy for w in tqdm(
    signal_windows,
    total=int(len(samples) / float(step_size))
))

window_silence = (e > silence_threshold for e in window_energy)

cut_times = (r * step_duration for r in rising_edges(window_silence))


# This is the step that takes long, since we force the generators to run.
print "Finding silences..."
cut_samples = [int(t * sample_rate) for t in cut_times]
cut_samples.append(-1)

cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in xrange(len(cut_samples) - 1)]


output_dir="/Users/jeff/Documents/increase_prior/recordings/split"
output_filename_prefix="a_001"
dry_run=0


for i, start, stop in tqdm(cut_ranges):
    output_file_path = "{}_{:03d}.wav".format(
        os.path.join(output_dir, output_filename_prefix),
        i
    )
    if not dry_run:
        print "Writing file {}".format(output_file_path)
        wavfile.write(
            filename=output_file_path,
            rate=sample_rate,
            data=samples[start:stop]
        )
    else:
        print "Not writing file {}".format(output_file_path)





