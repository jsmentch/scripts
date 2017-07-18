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
parser.add_argument('--indir', type=str, default='/Users/jeff/Documents/increase_prior/recordings/original', help='The input directory (with .wav files), default is /Users/jeff/Documents/increase_prior/recordings/original')
parser.add_argument('--outdir', type=str, default='/Users/jeff/Documents/increase_prior/recordings/split', help='The output dir. Defaults to /Users/jeff/Documents/increase_prior/recordings/split')
parser.add_argument('--min-silence-length',type=float, default=0.5, help='Minimum length of silence for a split. Defaults to 0.5 seconds.')
parser.add_argument('--silence-threshold', type=float, default=1e-6, help='energy level (between 0.0 and 1.0) below which the signal is regarded as silent. Defaults to 1e-6 == 0.0001%.')
parser.add_argument('--step-duration', type=float, default=None, help='amount of time to step through input file after calculating e. Smaller = slower, more accurate; Larger = faster, might miss. Default is min_silence_length/10')
parser.add_argument('--dry-run', '-d', action='store_true', help='Don\'t actually write any output files.')

parser.add_argument('--splitting', '-s', action='store_true', help='Don\'t split files and save them, skip to downbeat tracking, ramping (sets sr to 44100 for now)')

parser.add_argument('--downbeat-track', '-t', action='store_true', help='Don\'t run second step - RNN downbeat tracking with madmom, ramps.')
parser.add_argument('--outdir2', type=str, default='/Users/jeff/Documents/increase_prior/recordings/processed', help='The output dir for part to - downbeat tracking etc. Default is /processed')

args = parser.parse_args()

#variables for part 1
indir = args.indir
outdir = args.outdir
window_duration=args.min_silence_length
silence_threshold = args.silence_threshold
if args.step_duration is None:
	step_duration=args.min_silence_length/10 #
else:
    step_duration = args.step_duration
dry_run = args.dry_run
splitting= args.splitting
#variables for part 2
downbeat_track=args.downbeat_track
outdir2=args.outdir2

#if -s passed then skip to downbeat tracking
if not splitting:
	print "Splitting where energy is below {}% for longer than {}s.".format(
	    silence_threshold * 100.,
	    window_duration
	)

	#create list of .wav files from the indir
	file_list = glob.glob(os.path.join(indir, '*.wav'))
	#loops through files in the indir
	for f in file_list:
		f_base=os.path.basename(f)
		f_name, f_extension = os.path.splitext(f_base)
		
		print "Splitting file : %s"%f_base
		#load file
		sample_rate, samples = wavfile.read(filename=f, mmap=True)

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

		#filter out selected sections shorter than 30s
		cut_samples_filtered = []
		for t in xrange(len(cut_samples)-1):
		    if cut_samples[t+1] - cut_samples[t] > 30*sample_rate:
		        cut_samples_filtered.append(cut_samples[t])

		cut_samples_filtered.append(-1)
		
		cut_ranges = [(i, cut_samples_filtered[i], cut_samples_filtered[i+1]) for i in xrange(len(cut_samples_filtered) - 1)]

		for i, start, stop in tqdm(cut_ranges):
		    output_file_path = "%s_%03d.wav"%(
		        os.path.join(outdir, f_name),
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

else:
	print "not splitting, skipping to downbeat tracking"
	sample_rate=44100


#Downbeat tracking and ramp
if not downbeat_track:
	#create 50ms quartersine ramp up and ramp down
	sample=sample_rate/20 # of samples in 50 ms
	rampup=[0]*sample
	ramp = []
	for n in range(sample):
	    #50ms rampup
	    rampup[n]=sin(.5*pi*n/sample)

	    #ramp 'mask' of volume envelope to multiply with 6s+ clip
	    ramp = np.ones(int(6.315827664399093*sample_rate))
	    
	    #set beginning and of ramp mask to the ramp up and ramp down
	    ramp[0:2205] = rampup
	    #according to stack exchange this is a faster reverse than slicing the array yet less readable
	    ramp[-2205:] = list(reversed(rampup))
	    ramp = ramp.reshape(278528,1)
	    ramp = np.append(ramp, ramp, axis=1)
	    
	    
	def ramp_clip(audioin_6s):
	    audioin_6s=audioin_6s*ramp
	    return audioin_6s
	

	rnndb = madmom.features.beats.RNNDownBeatProcessor()
	

	#list the files of the previous outdir (split files) to loop over
	file_list2 = glob.glob(os.path.join(outdir, '*.wav'))

	for f2 in tqdm(file_list2):
		beats_dbeats = rnndb(f2)

		f2_base=os.path.basename(f2)
		f2_name, f2_extension = os.path.splitext(f2_base)

		audioin, sr, fmt = wavread(f2) #load .wav
		third_length=len(audioin)/441/3 #length of one third of the song in RNN windows (for 10 ms window)
		starttime = []
		audioin_6s_ramped=[]
		for i in range(3): #3
		    #search for downbeats from each third of the track from 7 seconds after the start of the third to 7 s before the end
		    third_start=(third_length*i)+700
		    third_end=third_length*(i+1)-700
		    #starttime given in 10 ms increments (window from RNN downbeat processor)
		    #argmax finds most probable downbeat from third section
		    starttime = (i*third_length)+np.argmax(beats_dbeats[third_start:third_end,1])
		    #starttime conversion to samples using sr
		    starttime=starttime*sr/100
		    
		    #a little over 6 seconds, matching lengths of stimulus files
		    audioin_6s = audioin[starttime:starttime+int(6.315827664399093*sr)]
		    
		    audioin_6s_ramped = audioin_6s * ramp
		    
		    f2_basename = os.path.basename(f2)
		    f2_basename, extension = os.path.splitext(f2_basename)
		    wavwrite(audioin_6s_ramped, os.path.join(outdir2, f2_basename + '_' + "%s" % i + ".wav"), fs=sr, enc="pcm16")
		    print "Processed file %s_%s" % (f2, i)

else:
	print "Not running downbeat-detection"


