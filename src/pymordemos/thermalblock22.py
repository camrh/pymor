#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Thermalblock demo.

Usage:
  thermalblock.py [-ehp] [--estimator-norm=NORM] [--extension-alg=ALG] [--grid=NI] [--help]
                  [--plot-solutions] [--plot-error-sequence] [--reductor=RED] [--test=COUNT]
                  XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  -e, --with-estimator   Use error estimator.

  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: trivial].

  --extension-alg=ALG    Basis extension algorithm (trivial, gram_schmidt, h1_gram_schmidt)
                         to be used [default: h1_gram_schmidt].

  --grid=NI              Use grid with 2*NI*NI elements [default: 100].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2
import numpy as np
from pymor.algorithms import greedy, trivial_basis_extension, gram_schmidt_basis_extension
#from pymor.analyticalproblems import ThermalRadialBlockProblem
from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors import reduce_to_subbasis
from pymor.reductors.linear import reduce_stationary_affine_linear
core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')
core.getLogger('pymor.la').setLevel('INFO')


def thermalblock_demo(args):
    args['XBLOCKS'] = int(args['XBLOCKS'])
    args['YBLOCKS'] = int(args['YBLOCKS'])
    args['--grid'] = int(args['--grid'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt', 'h1_gram_schmidt'}

    print('Solving on TriaGrid(({0},{0}))'.format(args['--grid']))

    print('Setup Problem ...')
    problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'],args['YBLOCKS']))

    print('Discretize ...')
    discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / args['--grid'])

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        discretization.visualize(Us, legend=legend, title='Detailed Solutions for different parameters', block=True)

    print('RB generation ...')

    error_product = discretization.h1_product if args['--estimator-norm'] == 'h1' else None
    reductor = partial(reduce_stationary_affine_linear, error_product=error_product)
    extension_algorithms = {'trivial': trivial_basis_extension,
                            'gram_schmidt': gram_schmidt_basis_extension,
                            'h1_gram_schmidt': partial(gram_schmidt_basis_extension, product=discretization.h1_product)}
    extension_algorithm = extension_algorithms[args['--extension-alg']]
    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         use_estimator=args['--with-estimator'], error_norm=discretization.h1_norm,
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']

    print('\nSearching for maximum error on random snapshots ...')

    def error_analysis(d, rd, rc, mus):
        print('N = {}: '.format(rd.operator.dim_source), end='')
        h1_err_max = -1
        h1_est_max = -1
        cond_max = -1
        for mu in mus:
            print('.', end='')
            u = rd.solve(mu)
            URB = rc.reconstruct(u)
            U = d.solve(mu)
            h1_err = d.h1_norm(U - URB)[0]
            h1_est = rd.estimate(u, mu=mu)
            cond = np.linalg.cond(rd.operator.assemble(mu)._matrix)
            if h1_err > h1_err_max:
                h1_err_max = h1_err
                mumax = mu
            if h1_est > h1_est_max:
                h1_est_max = h1_est
                mu_est_max = mu
            if cond > cond_max:
                cond_max = cond
                cond_max_mu = mu
        print()
        return h1_err_max, mumax, h1_est_max, mu_est_max, cond_max, cond_max_mu

    tic = time.time()

    real_rb_size = len(greedy_data['basis'])
    if args['--plot-error-sequence']:
        N_count = min(real_rb_size - 1, 25)
        Ns = np.linspace(1, real_rb_size, N_count).astype(np.int)
    else:
        Ns = np.array([real_rb_size])
    rd_rcs = [reduce_to_subbasis(rb_discretization, N, reconstructor)[:2] for N in Ns]
    mus = list(discretization.parameter_space.sample_randomly(args['--test']))

    errs, err_mus, ests, est_mus, conds, cond_mus = zip(*(error_analysis(discretization, rd, rc, mus)
                                                        for rd, rc in rd_rcs))
    h1_err_max = errs[-1]
    mumax = err_mus[-1]
    cond_max = conds[-1]
    cond_max_mu = cond_mus[-1]
    h1_ests_max = ests[-1]
    toc = time.time()
    t_est = toc - tic

    print('''
    *** RESULTS ***

    Problem:
       number of blocks:                   {args[XBLOCKS]}x{args[YBLOCKS]}
       h:                                  sqrt(2)/{args[--grid]}

    Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}^({args[XBLOCKS]}x{args[YBLOCKS]})
       used estimator:                     {args[--with-estimator]}
       estimator norm:                     {args[--estimator-norm]}
       extension method:                   {args[--extension-alg]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal H1-error:                   {h1_err_max}  (mu = {mumax})
       maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()

    if args['--plot-error-sequence']:
        plt.semilogy(Ns, errs, Ns, ests)
        plt.legend(('error', 'estimator'))
        plt.show()
    #if 1:
    #    U = discretization.solve(mumax)
    #    #np.savetxt("data.csv", delimiter=",")
    if args['--plot-err']:
        U = discretization.solve(mumax)
        URB = reconstructor.reconstruct(rb_discretization.solve(mumax))
        print( U)
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution', separate_colorbars=True, block=True)

    return h1_err_max, mumax, cond_max, cond_max_mu, t_est, h1_ests_max

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    args['XBLOCKS'] = 2
    args['YBLOCKS'] = 2
    #args['SNAPSHOTS'] 
    # run demo
    #for sn in range(2,3):
    sn = 2;
    basis = []
    h1_err = []
    co = []
    t = []
    h1_est = []
    filename = ('x2y2-h1-%03d.csv' % int(sn))
    with open(filename, 'wb') as f:
        f.write(b"base, h1err, cond, eff, h1estmax\n")

    for i in range(1,35):
        basis.append( int(i))
        args['RBSIZE'] = i
        args['SNAPSHOTS'] = sn
        h1_err_max, numax, cond_max, cond_max_mu, t_est, h1_ests_max= thermalblock_demo(args)
        t.append(t_est)
        h1_err.append(h1_err_max)
        co.append(cond_max)
        h1_est.append(h1_ests_max)
        #arr = zip(i, h1_err_max, cond_max, t, h1_ests_max)

        with open(filename,'a+b') as f:
            f.write('{0}, {1}, {2}, {3}, {4}\n'.format(i, h1_err_max, cond_max, t_est, h1_ests_max[0]))
            #np.savetxt(f, arr, delimiter=",")
                  
    #arr = zip(basis, h1_err, co, t, h1_est)
    #with open(filename, 'wb') as f:
    #f.write("base, h1err, cond, eff, h1estmax\n")
    #np.savetxt(f, arr, delimiter=",")
