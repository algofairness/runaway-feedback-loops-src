#!/usr/bin/env python3
import math
import pylab
import sys
import click
from polya import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def histogram(urn, ndraws, nrunsm, lambdaa, lambdab):
    x = []
    for i in range(nruns):
        urn.reset()
        for _ in urn.draw(ndraws): pass
        x.append(urn.state[0] / (urn.state[0] + urn.state[1]))
        if i % 100 == 0:
            print(".", file=sys.stderr, end='')
            sys.stderr.flush()
    print("", file=sys.stderr)
    pylab.hist(x)

def singlerun(urn, ndraws, nrunsm, lambdaa, lambdab):
    x = []
    y = []
    for state in urn.draw(ndraws):
        x.append(state[0])
        y.append(state[1])
    pylab.plot(x, 'r')
    pylab.plot(y, 'k')
    pylab.plot(list(a + b for a,b in zip(x,y)), 'b--')

def singleprob(urn, ndraws, nrunsm, lambdaa, lambdab):
    x = []
    y = []
    s = 0
    c = 0
    for state in urn.draw(ndraws):
        x.append(state[0])
        y.append(state[1])
        s += (x[-1] / (x[-1]+y[-1]))
        c += 1
    pylab.plot(list((a / (a + b)) for (a,b) in zip(x, y)), 'r')
    print(s/c)
    print(urn.state)

def probplot(urn, ndraws, nruns, lambdaa, lambdab):
    burn_in = 0
    histo_resolution = 100
    counts = numpy.zeros((histo_resolution+1, ndraws-burn_in))
    for i in range(nruns):
        urn.reset()
        for _ in urn.draw(burn_in): pass
        for j, state in enumerate(urn.draw(ndraws-burn_in)):
            u = state[0] / (state[0] + state[1])
            counts[100-int(u*histo_resolution), j] += 1
        if i % 100 == 0:
            print(".", file=sys.stderr, end='')
            sys.stderr.flush()
    v = (counts * numpy.linspace(1, 0, 101)[:,numpy.newaxis]).sum(axis=0) / counts.sum(axis=0)
    counts_max = counts.max(axis=0)
    counts = counts / counts_max
    pylab.imshow(counts, extent=[0, counts.shape[1],
                                 0, 1.01],
                 aspect=ndraws, cmap=pylab.cm.gray_r)
    ((a, b), (c, d)) = urn.update_matrix
    result = numpy.roots([c+d-a-b, a - 2 * c - d, c])
    for r in result:
        if r >= 0 and r <= 1:
            pylab.plot([0, counts.shape[1]], [r, r], 'r--')
#    pylab.plot([0, counts.shape[1]], [(a + b) / (a + b + c + d),
#                                      (a + b) / (a + b + c + d)], 'g--')
    pylab.plot([0, counts.shape[1]], [ lambdaa / (lambdab + lambdaa), lambdaa / (lambdab + lambdaa)], 'g--')
    pylab.plot(numpy.arange(counts.shape[1]), v, 'y-')
    pylab.xlim([0, counts.shape[1]])
    pylab.ylim([1, 0])
    print(result)
    pylab.colorbar()
    
commands = {
    "singlerun": singlerun,
    "histogram": histogram,
    "probplot": probplot,
    "singleprob": singleprob
    }

urn_class = PolyaUrn

def exponential_decay_option(click, param, exponential_decay):
    global urn_class
    if exponential_decay == 0.0:
        return
    print("Setting exponential decay to %f" % exponential_decay)
    urn_class = add_exponential_decay(urn_class, exponential_decay)

def linear_surprise_option(click, param, linear_surprise):
    global urn_class
    if not linear_surprise:
        return
    print("Setting urn to behave with linear surprise")
    urn_class = add_linear_surprise(urn_class)

def partial_surprise_option(click, param, partial_surprise):
    global urn_class
    if partial_surprise == (None, None):
        return
    print("Setting urn to behave with partial surprise")
    urn_class = add_partial_surprise(urn_class, partial_surprise)

def weighted_surprise_option(click, param, weighted_surprise):
    global urn_class
    if weighted_surprise == (None, None, None, None):
        return
    print("Setting urn to behave with weighted surprise")
    urn_class = add_weighted_surprise(urn_class, weighted_surprise)

def sqrt_surprise_option(click, param, sqrt_surprise):
    global urn_class
    if not sqrt_surprise:
        return
    print("Setting urn to behave with sqrt surprise")
    urn_class = add_sqrt_surprise(urn_class)

def poisson_option(click, param, poisson):
    global urn_class
    if not poisson:
        return
    print("Setting urn to be Poisson")
    urn_class = add_poisson_update(urn_class)

def truncation_option(click, param, truncation):
    global urn_class
    if truncation is None:
        return
    print("Truncating urn update to %d" % truncation)
    urn_class = add_truncation(urn_class, truncation)

urn_params = [[1.0,0.0],[0.0,1.0]]

def mixed_option(click, param, mixed):
    if mixed == (None, None, None, None):
        return
    (d_a, d_b, r_a, r_b) = mixed
    urn_params[0][0] = d_a + r_a
    urn_params[0][1] = r_b
    urn_params[1][0] = r_a
    urn_params[1][1] = d_b + r_b

def set_a(click, param, a):
    if a is None: return
    urn_params[0][0] = a
def set_b(click, param, b):
    if b is None: return
    urn_params[0][1] = b
def set_c(click, param, c):
    if c is None: return
    urn_params[1][0] = c
def set_d(click, param, d):
    if d is None: return
    urn_params[1][1] = d

command_docstring = """Commands:
 - singlerun
 - histogram
 - probplot
 - singleprob
"""

@click.command()
@click.option('--a', callback=set_a, type=float, help="Set the urn's a parameter")
@click.option('--b', callback=set_b, type=float, help="Set the urn's b parameter")
@click.option('--c', callback=set_c, type=float, help="Set the urn's c parameter")
@click.option('--d', callback=set_d, type=float, help="Set the urn's d parameter")
@click.option('--command', default="singlerun", help=command_docstring)
@click.option('--ndraws', type=int, default=2000, help="How many draws from an urn")
@click.option('--nruns', type=int, default=1000, help="How many urn runs to run")
@click.option('--nr', type=float, default=1, help="Number of red balls in urn's starting configuration")
@click.option('--nb', type=float, default=1, help="Number of black balls in urn's starting configuration")
@click.option('--lambdaa', type=float, default=0.5, help="Underlying true rate for neighborhood A (red balls)")
@click.option('--lambdab', type=float, default=0.5, help="Underlying true rate for neighborhood B (black balls)")
@click.option('--exponential_decay', callback=exponential_decay_option, type=float, default=0, help="Add exponential decay to the urn")
@click.option('--truncation', callback=truncation_option, type=int, help="Truncate the maximum number of new balls to add to urn")
@click.option('--linear_surprise', callback=linear_surprise_option, is_flag=True, help="Incorporate a linear surprise factor in urn update")
@click.option('--partial_surprise', callback=partial_surprise_option, nargs=2, type=(float, float), default=(None, None), help="Incorporate a partial surprise factor in urn update, adding only reported crimes")
@click.option('--weighted_surprise', callback=weighted_surprise_option, nargs=4, type=(float, float, float, float), default=(None, None, None, None), help="Incorporate a weighted surprise factor in urn update, adding only reported crimes")
@click.option('--sqrt_surprise', callback=sqrt_surprise_option, is_flag=True, help="Incorporate a sqrt surprise factor in urn update")
@click.option('--poisson', callback=poisson_option, is_flag=True, help="urn updates are draws from a poisson instead of deterministic")
@click.option('--mixed', callback=mixed_option, nargs=4, type=(float, float, float, float), default=(None, None, None, None), help="set parameters of a mixed urn d_A, d_B, r_B, r_B")
@click.option('--interactive', is_flag=True, help='if set, show image interactively instead of saving to file')
@click.option('--output', type=str, default="fig_out.png", help='name of output file if noninteractive')
def main(a, b, c, d, output, command, ndraws, nruns, nr, nb, lambdaa, lambdab, exponential_decay, truncation, interactive, linear_surprise, sqrt_surprise, partial_surprise, weighted_surprise, poisson, mixed):
    print("Urn starting state: %s" % ((nr, nb),))
    print("Urn parameters: %s" % urn_params)
    urn = urn_class((nr, nb), urn_params)
    commands[command](urn, ndraws, nruns, lambdaa, lambdab)
    if interactive:
        pylab.show()
    else:
        pylab.savefig(output)

if __name__ == "__main__":
    main()
