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

def maximas(ar,threshold):
    derleft = numpy.diff(ar)
    derup = numpy.diff(ar,axis=0)
    derright = numpy.hstack((derleft,numpy.zeros((derleft.shape[0],1))))
    derleft = numpy.hstack((numpy.zeros((derleft.shape[0],1)),derleft))
    derdown = numpy.vstack((derup,numpy.zeros((1,derup.shape[1]))))
    derup = numpy.vstack((numpy.zeros((1,derup.shape[1])),derup))
    mx = (derleft>=threshold)&(derright<=threshold)&(derup>=threshold)&(derdown<=threshold)
    return mx

def seperatemaximas(ar,xwidth,ywidth):
    mx = maximas(ar,0)
    indices = numpy.indices(ar.shape)[:,mx]
    mxims = ar[mx].argsort()
    for maxima in reversed(mxims):
        i,j = indices[:,maxima]
        m = ar[i,j]
        if m>0:
            ar[max(0,i-ywidth):i+ywidth,max(0,j-xwidth):j+xwidth] = 0
            ar[i,j] = m
    ar[mx==False]=0
    ar[ar>0] = 1
    return ar

def makepairs(ar):
    mx = ar==1
    indices = numpy.indices(ar.shape)[:,mx].transpose()
    pairs = []
    for i,j in indices:
        tmpboard = mx[i+1:i+35,j+1:j+45]
        print "-"
        try:
            tmpindices = numpy.indices(tmpboard.shape)[:,tmpboard].transpose()
            for ii,jj in tmpindices:
                pairs.append([j,i,i+ii+1,jj+1])#[time,f1,f2,dt]
                ar = makeline(ar,i,j,i+ii+1,j+jj+1)
        except:pass
    return numpy.array(sorted(pairs,key=lambda x:x[0]))


def makeline(ar,x0,y0,x1,y1):
    dx = x1-x0
    dy = y1-y0
    slope = float(dy)/dx
    if abs(slope)<1:
        for x in range(dx):
            ar[x0+x,y0+round(slope*x)] = 0.3
    else:
        for y in range(dy):
            ar[x0+round(y/slope),y0+y] = 0.3
    return ar


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

    ffts = ffts[:,400:]
    print ffts.shape
    print numpy.amax(ffts)
    # go to log domain (with lower bound = max / 1e6)
    fftlog = numpy.transpose(numpy.log(numpy.maximum(ffts, numpy.amax(ffts)/1e6)))

    # samples = numpy.array([f for time,f in sorted(analyzer.samples.iteritems())])



    ffts = numpy.transpose(ffts)
    mxms = seperatemaximas(fftlog.copy(),15,20)
    #mxms = makeline(mxms,10,10,20,50)
    #mxms = makeline(mxms,20,50,50,75)
    pairs = makepairs(mxms)
    print pairs
    print len(pairs)
    pp.figure()
    pp.imshow(fftlog+mxms*15)
    pp.gray()
    # pp.figure()
    # pp.imshow(samples)
    interactive_shell(locals(), globals())