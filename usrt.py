#!/usr/bin/python

# pts =  presentation time stamp in time_base units

def interactive_shell(_locals, _globals):
    try:
        from IPython.Shell import IPShellEmbed
        ipshell = IPShellEmbed(())
        ipshell(local_ns = _locals, global_ns=_globals)
    except ImportError:
        import code
        code.interact(local=_locals)

import sys
import pyffmpeg
import numpy
from numpy.fft import fft
from matplotlib import pyplot as pp
pp.ion()


#frate=44100.
frate=16000.
freq=8
df=2048
do=df-(df/freq)
di=df-do
nx=df//di

# TS_AUDIO={ 'audio1':(1, -1, {'hardware_queue_len':1000, 'dest_frame_size':df, 'dest_frame_overlap':do} )}
TS_AUDIO={ 'audio1':(1, -1, {'hardware_queue_len':1000, 'dest_frame_size':512, 'dest_frame_overlap':256} )}

# TS_AUDIO={ 'audio1':(1, -1, {'hardware_queue_len':1000} )}

class AudioAnalyzer(object):
    def __init__(self):
        self.ffts = {}
        self.samples = {}
        self.cnt = 0
        self.lens = []

    def read_audio(self, (sample, pts, fps)):
        # interactive_shell(_locals = locals(), _globals=globals())    
        # _ = numpy.argmax(abs(fft(sample[:,0])))
        self.lens.append(len(sample))
        self.ffts[pts] = abs(fft(sample[:,0]))
        self.samples[pts] = sample[:,0]
        self.cnt+=1
        if (int(pts) % 5) == 0:
            print `pts` + "\r",

if __name__ == '__main__':
    filename = sys.argv[1]
    mp = pyffmpeg.FFMpegReader()
    mp.open(filename, track_selector=TS_AUDIO)
    # video, audio = mp.get_tracks()
    [audio] = mp.get_tracks()
    analyzer = AudioAnalyzer()
    audio.set_observer(analyzer.read_audio)
    # audio.seek_to_seconds(mp.duration_time() - 10)
    # audio.seek_to_seconds(5)
    for frame in xrange(6*24):
        # video.get_next_frame()
        audio.get_next_frame()
    
    ffts = numpy.array([f[1:-1] for time,f in sorted(analyzer.ffts.iteritems())])
    # go to log domain (with lower bound = max / 1e6)
    fftlog = numpy.log(numpy.maximum(ffts, numpy.amax(ffts)/1e6))

    # samples = numpy.array([f for time,f in sorted(analyzer.samples.iteritems())])
    
    pp.figure()
    pp.imshow(numpy.transpose(fftlog))
    pp.gray()
    # pp.figure()
    # pp.imshow(samples)
    interactive_shell(locals(), globals())

