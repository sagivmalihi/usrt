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
import pickle
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

def makepairs(fftar):
    ar = fftar.copy()
    ar = seperatemaximas(ar,15,20)
    mx = ar==1
    indices = numpy.array(numpy.where(mx)).transpose()
    pairs = []
    for i,j in indices:
        tmpboard = mx[i+1:i+35,j+1:j+45]
        try:
            tmpindices = numpy.array(numpy.where(tmpboard)).transpose()
            for ii,jj in tmpindices:
                pairs.append([j,i,i+ii+1,jj+1])#[time,f1,f2,dt]
                ar = makeline(ar,i,j,i+ii+1,j+jj+1)
        except:pass
    return numpy.array(sorted(pairs,key=lambda x:x[0])),ar


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
    if len(sys.argv)>2:
        creating = sys.argv[2]=="create"
    else:
        creating = False
    mp = pyffmpeg.FFMpegReader()
    mp.open(filename, track_selector=TS_AUDIO)
    # video, audio = mp.get_tracks()
    [audio] = mp.get_tracks()
    analyzer = AudioAnalyzer()
    audio.set_observer(analyzer.read_audio)
    # audio.seek_to_seconds(mp.duration_time() - 10)
    if creating:
        while True:
            try:
                audio.get_next_frame()
            except IOError:
                break
    else:
        audio.seek_to_seconds(3)
        for frame in xrange(5*24):
            try:
                audio.get_next_frame()
            except IOError:
                break

    ffts = numpy.array([f[1:-1] for time,f in sorted(analyzer.ffts.iteritems())])

    ffts = ffts[:,400:]
    # go to log domain (with lower bound = max / 1e6)
    fftlog = numpy.transpose(numpy.log(numpy.maximum(ffts, numpy.amax(ffts)/1e6)))

    # samples = numpy.array([f for time,f in sorted(analyzer.samples.iteritems())])



    ffts = numpy.transpose(ffts)
    #mxms = makeline(mxms,10,10,20,50)
    #mxms = makeline(mxms,20,50,50,75)
    pairs,ar = makepairs(ffts)
    """
    print pairs
    print len(pairs)
    pp.figure()
    pp.imshow(fftlog+ar*3)
    pp.gray()
    """
    # pp.figure()
    # pp.imshow(samples)

    if creating:
        pairs.dump("data.txt")
        print "data saved"
        sys.exit(0)

    #now search in data
    data = numpy.load("data.txt")
    matching = []
    sdata = data[:,1:]
    for p in pairs:
            row = numpy.all( sdata==p[1:] ,1)
            for i in numpy.where(row)[0]:
                # a match !
                matching.append(data[i,0]-p[0])
    histogram = numpy.histogram(matching,numpy.unique(matching))
    argmax = histogram[0].argmax()
    if histogram[0][argmax]>50:
        offset = histogram[1][argmax]
        print "----"
        print "offset:",
        print offset
        print "in seconds :",
        print offset/(24*7)
        print "----"

    else: #no match :(
        offset=None
        print "no offset found "
        print "---"

    #interactive_shell(locals(), globals())