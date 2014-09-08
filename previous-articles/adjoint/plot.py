from __future__ import unicode_literals
import os

os.environ['PATH']+= ":/usr/texbin"
import json
import pylab
import numpy
import matplotlib
from itertools import cycle

filetype = 'png'
time_units = 'minutes'
time_division = 60.0
length_units = 'miles'
cb_y = .3
line_width= .000
dpi = 500



def add_ramps():
  # [pylab.axhspan(orpair[0],orpair[1], fill=False, linestyle='solid', linewidth=line_width) for orpair in orpairs]
  for i, orpair in enumerate(orpairs[1:]):
    if i is 6:
      continue
    if i is 7:
      s = 'ramp 7,8'
    else:
      s = 'ramp {0}'.format(i + 1)
    pylab.text(80,orpair[1],s, fontsize='7')

def add_colorbars(data):
  cax = pylab.axes([0.8, 0.3, 0.02, 0.6])
  pylab.colorbar(cax=cax)
  # cax = pylab.axes([0.835, cb_y, 0.02, 0.13])
  # lims = (data.min().min(), data.max().max())  
  # ticks = [int(a) for a in pylab.linspace(lims[0], lims[1], 3)]
  # print ticks
  # pylab.colorbar(cax = cax, ticks= ticks)
  # pylab.colorbar(ticks= ticks)



def diff_figure(data):
  lims = (data.min().min(), data.max().max())
  frac_switch = lims[0] * -1 / (lims[1] - lims[0])
  error = .05
  if lims[0] >= 0:
    red = 1.0
  else:
    red = 0.0
  cmap = {'blue':   [(0.0,  red, red),
                   (frac_switch * (1 - error),  1.0, 1.0),
                   (frac_switch * (1 + error),  1.0, 1.0),
                   (1.0,  1.0, 1.0)],
         'green': [(0.0,  red, red),
                   (frac_switch * (1 - error),  1.0, 1.0),
                   (frac_switch * (1 + error),  1.0, 1.0),
                   (1.0,  0.0, 0.0)],
         'red':  [(0.0,  1.0, 1.0),
         (frac_switch * (1 - error),  1.0, 1.0),
                   (frac_switch * (1 + error),  1.0, 1.0),
                   (1.0,  0.0, 0.0)]}
  colormap = matplotlib.colors.LinearSegmentedColormap("test", cmap)
  newfigure()
  plotSpaceTime(data,cmap=colormap)


def saveto(fn, ft = filetype):
  pylab.savefig('/Users/jdr/Documents/github/AdjointPaper/restart/images/{0}.{1}'.format(fn, ft), dpi=dpi)


class LineCycler(object):
  """docstring for LineCycler"""

  @classmethod
  def reset(cls):
    cls.lines = cycle(["-","--","-.",":"])

  @classmethod
  def next(cls):
    try:
      return cls.lines.next()
    except:
      cls.reset()
      return cls.next()

def newfigure():
  LineCycler.reset()
  fig = pylab.figure()
  pylab.clf()
  pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
  return fig

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (pylab.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)

base_dir = '/Users/jdr/Documents/github/web-ramp/'
fnA = base_dir +  'ExperimentA.json'
fnB = base_dir +  'ExperimentB.json'
fnC = base_dir +  'ExperimentC.json'
fnIter = base_dir + 'iterspeed.json'

with open(fnA, "r") as fn:
  data = json.load(fn)

nc_density = numpy.clip(pylab.array(data['experiment']['nc']['density']), 0, 1)
nc_queue = numpy.clip(pylab.array(data['experiment']['nc']['queue']), 0, 10000000)
diff_density = numpy.clip(pylab.array(data['experiment']['adjoint']['densityDiff']), -1, 1)
print 'diff_density', diff_density.max().max(), diff_density.min().min()
diff_queue = pylab.array(data['experiment']['adjoint']['queueDiff'])
print 'diff_queue', diff_queue.max().max(),  diff_queue.min().min()

length = data['length']
time = data['time'] / time_division
lengths = slice(0, length, length / len(nc_queue[0]))
times = slice(0, time, time / len(nc_queue))
TT,XX = pylab.mgrid[times, lengths]


# onramps = data['onramps']
# better onramp approach
onramps = [i for i, xxx in enumerate([sum([abs(xx) for xx in row]) for row in nc_queue.T]) if xxx > 0]
orpairs = [(XX[0][i], XX[0][i+1]) for i in onramps]
print 'orpairs', orpairs

tttNC = data['experiment']['nc']['ttt']

print tttNC
print 'loaded exp a'
totalCongestion = data['experiment']['totalCongestion']
totalFreeFlow = data['experiment']['totalFreeFlow']


def plotSpaceTime(d, cmap = 'binary'):
  # pylab.gca().pcolorfast(t,x,d, cmap=cmap)
  pylab.pcolor(TT,XX,d, cmap=cmap)
  pylab.xlabel(r"\textit{Time} (%s)" % time_units)
  pylab.ylabel(r"\textit{Freeway offset} (%s)" % length_units)


def congestionDecrease(ttt, ff = totalFreeFlow, c = totalCongestion):
  return 100 * (1 - (ttt - ff) / c)

adjoint_ttt = data['experiment']['onerun']['adjoint']
alinea_ttt = data['experiment']['onerun']['alinea']

running_time = map(list,zip(*data['experiment']['adjoint']['runtime']))


newfigure()
max_rt = 2 * pylab.median(pylab.diff([0] + running_time[0]))
running_time[0] = pylab.cumsum([min(max_rt, rt) for rt in pylab.diff([0] + running_time[0])])

pylab.plot([0] + [x / 1000. for x in running_time[0]], [0] + [congestionDecrease(a) for a in running_time[1]], LineCycler.next(), label="Adjoint")
pylab.hold(True)
pylab.plot([0] + [x / 1000. for x in running_time[0]], [congestionDecrease(alinea_ttt)] * (1 + len(running_time[0])), LineCycler.next(), label="Alinea")
pylab.xlabel(r"\textit{Running time} (seconds)")
pylab.ylabel(r"\textit{Reduced Congestion} (\%)")
pylab.legend(loc='lower right')
print 'before save'
saveto('longsim', 'pdf')

print 'running time done'

print 'nc density and queue computed'
print nc_density.shape
print nc_queue.shape

newfigure()
plotSpaceTime(nc_density)
cax = pylab.axes([0.8, 0.3, 0.02, 0.6])
pylab.colorbar(cax=cax)
saveto('ncdensity')
newfigure()
plotSpaceTime(nc_queue)
add_ramps()
add_colorbars(nc_queue)
saveto('ncqueue')


diff_figure(diff_density)
cax = pylab.axes([0.8, 0.3, 0.02, 0.6])
pylab.colorbar(cax=cax)
saveto('densdiff')
diff_figure(diff_queue)
add_ramps()
add_colorbars(diff_queue)
saveto('queuediff')

newfigure()

with open(fnC, "r") as fn:
  data = json.load(fn)
print data
adjoint_ttt_noise = data['experiment']['mpc']['adjoint']
alinea_ttt_noise = data['experiment']['mpc']['alinea']

cong_compares = [congestionDecrease(a) for a in [adjoint_ttt, adjoint_ttt_noise, alinea_ttt, alinea_ttt_noise]]
cong_labels =['Adjoint', 'Adjoint w/ Noise', 'Alinea', 'Alinea w/ Noise']
x_vals = pylab.arange(len(cong_compares))
pylab.bar(x_vals, cong_compares)
pylab.xticks(x_vals + .5, cong_labels)
pylab.ylabel(r"\textit{Reduced Congestion} (\%)")
saveto('longmpc', 'pdf')


with open(fnB, "r") as fn:
  noise = json.load(fn)

totalFreeFlow = noise['ffCost']
totalCongestion = noise['congestedCost']

alinea_noise = noise['ttts']['alinea']
adjoint_noise = noise['ttts']['adjoint']

newfigure()
pylab.plot(noise['noises'], [congestionDecrease(a, totalFreeFlow, totalCongestion) for a in adjoint_noise], LineCycler.next(), label="Adjoint")
pylab.hold(True)
pylab.plot(noise['noises'], [congestionDecrease(a, totalFreeFlow, totalCongestion) for a in alinea_noise], LineCycler.next(), label="Alinea")
pylab.xscale('log')
pylab.legend()
pylab.xlabel(r"\textit{Noise factor} (-)")
pylab.ylabel(r"\textit{Reduced Congestion} (\%)", labelpad = -30)
saveto('noiseplot', 'pdf')

with open(fnIter, 'r') as fn:
  iterspeed = json.load(fn)

newfigure()
pylab.semilogx(iterspeed['slow'][0],iterspeed['slow'][1], LineCycler.next(),label='Finite differences')
pylab.hold(True)
pylab.semilogx(iterspeed['fast'][0],iterspeed['fast'][1], LineCycler.next(),label='Adjoint')
pylab.xlabel(r"\textit{Running time} (ms)")
pylab.ylabel(r"\textit{Total travel time} (veh-s)", labelpad=-30)
pylab.legend()
saveto('itergrad', 'pdf')