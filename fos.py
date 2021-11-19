# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:49:15 2021

@author: gawe
"""
import numpy as _np
import itertools
import time

# import math
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt


"""
 usage: 1.model = FOSModel(max_delay_in_input, max_delay_in_output, max_order, max_m, mse_reduction_threshold)
        2.model.fit(x, y)
        3.model.predict(x, y) # x is longer than y
"""


class FOSModel:
    """
    Implementation of Fast Orthogonal Search algorithm

    FOS is used to identify an ARMA model. reference: Applications of fast orthogonal search: Time-series analysis and resolution of signals in noise

    usage:

        1.model = FOSModel(max_delay_in_input, max_delay_in_output,
                           max_order, max_m, mse_reduction_threshold),
            where max_m is the max number of terms,
            mse_reduction_threshold is the early-stop criteria

        2.model.fit(x, y),
            x (input) and y (output) are 1-dimentional series

        3.model.predict(x, y),
            all x points and a few initial points of y, predicted y series will be returned


    Korenberg, M.J., Paarmann, L.D. Applications of fast orthogonal search:
    Time-series analysis and resolution of signals in noise.
    Ann Biomed Eng 17, 219–231 (1989). https://doi.org/10.1007/BF02368043

    @article{Korenberg:1989:ROA:2733743.2733908,
     author = {Korenberg, M. J.},
     title = {A Robust Orthogonal Algorithm for System Identification and Time-series Analysis},
     journal = {Biol. Cybern.},
     issue_date = {February  1989},
     volume = {60},
     number = {4},
     month = feb,
     year = {1989},
     issn = {0340-1200},
     pages = {267--276},
     numpages = {10},
     url = {http://dx.doi.org/10.1007/BF00204124},
     doi = {10.1007/BF00204124},
     acmid = {2733908},
     publisher = {Springer-Verlag New York, Inc.},
     address = {Secaucus, NJ, USA},
    }
    """
    DEBUG = False

    def __init__(self, max_delay_in_input, max_delay_in_output, max_order, max_m, mse_reduction_threshold, mse_threshold=0):

        if max_delay_in_input < 0 or max_delay_in_output < 0 or max_order < 1 or max_m < 1 or mse_reduction_threshold < 0 or mse_threshold < 0:
            raise ValueError("illegal parameter!")

        self.MAX_M = max_m
        self.MSE_THRESHOLD = mse_threshold
        self.MSE_REDUCTION_THRESHOLD = mse_reduction_threshold
        self.MAX_ORDER = max_order
        self.final_P = []
        self.final_P_in_polynomial_form = []
        self.final_a_m = []
        self.L = max_delay_in_input  # delay in x
        self.K = max_delay_in_output  # delay in y
        self.N_0 = max(max_delay_in_input, max_delay_in_output)
        self.candidate_list = []
        # generate candidate from order 1 to max_order
        for order in range(1, max_order + 1):
            self.candidate_list.extend(self.generate_candidate_list(order))

    def generate_candidate_list(self, sum_order):
        candidate_func_list = []
        for order in range(0, sum_order + 1):
            order_of_x_term = order
            order_of_y_term = sum_order - order
            comb_of_x = []
            comb_of_y = []
            together = []
            if order_of_x_term > 0:
                comb_of_x = itertools.combinations_with_replacement(range(0, self.L + 1), order_of_x_term)
            if order_of_y_term > 0:
                comb_of_y = itertools.combinations_with_replacement(range(1, self.K + 1), order_of_y_term)
            if order_of_x_term > 0 and order_of_y_term > 0:
                together = list(itertools.product(comb_of_x, comb_of_y))
            else:
                together = list(itertools.zip_longest(comb_of_x, comb_of_y))
            candidate_func_list.extend(together)
        return candidate_func_list

    def generate_P_func_from_delays(self, delay_of_x, delay_of_y, x, y):

        def P(n):
            if delay_of_x == None and delay_of_y == None:
                raise ValueError('order for P cannot be zero')
            p = 1
            if delay_of_x != None:
                for l in delay_of_x:
                    p = p * (x[n - l] if n >= l else 0)
            if delay_of_y != None:
                for k in delay_of_y:
                    p = p * (y[n - k] if n >= k else 0)
            return p

        return P

    def calculate_value_of_P(self, P):
        P_value = []
        for i in range(0, len(self.X)):
            if P == 1:
                P_value.append(1)
            else:
                P_value.append(P(i))
        return P_value

    def calculate_time_average_of_P(self, P_M, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n)
        return summ / (end - start)

    def calculate_time_average_of_P_square(self, P_M, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n) ** 2
        return summ / (end - start)

    def calculate_time_average_of_P_M_times_P_r(self, P_M, P_r, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n) * P_r(n)
        return summ / (end - start)

    def calculate_time_average_of_P_M_times_y(self, P_M, Y, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n) * Y[n]
        return summ / (end - start)

    def fit(self, X, Y):
        print("start fitting")
        start_time = time.time()

        if len(Y) != len(X):
            raise ValueError("length of input and output data must be equal")

        Y_bar = _np.mean(Y[self.N_0:])

        self.final_P = []
        self.final_P_in_polynomial_form = []
        self.final_a_m = []

        # init global storage
        P = [None] * (self.MAX_M + 1)
        P[0] = 1

        # another form of P
        P_in_polynomial_form = [None] * (self.MAX_M + 1)
        P_in_polynomial_form[0] = 1

        Alpha = [[0 for x in range(self.MAX_M + 1)] for y in range(self.MAX_M + 1)]

        D = [[0 for x in range(self.MAX_M + 1)] for y in range(self.MAX_M + 1)]
        D[0][0] = 1

        C = [None] * (self.MAX_M + 1)
        C[0] = Y_bar

        g = [None] * (self.MAX_M + 1)
        g[0] = Y_bar

        M = 1

        while (True):
            if FOSModel.DEBUG:
                print("\n################### choose candidate for M=", M, "###################")

            # init storage for current M
            MAX_Q_M = 0
            INDEX_OF_BEST_P_M = -1

            for index, candidate in enumerate(self.candidate_list):

                # init local storage for current candidate in current M
                D_M_r = [0] * (M + 1)
                alpha_M_r = [0] * (M + 1)

                delay_of_x = candidate[0]
                delay_of_y = candidate[1]
                if FOSModel.DEBUG:
                    print("\n----------------- evaluate the %dth candidate in the list -----------------" % (index))

                P_M = self.generate_P_func_from_delays(delay_of_x, delay_of_y, X, Y)

                P_bar = self.calculate_time_average_of_P(P_M, self.N_0, len(X))

                P_square_bar = self.calculate_time_average_of_P_square(P_M, self.N_0, len(X))
                if FOSModel.DEBUG:
                    print("delay_of_x: ", delay_of_x)
                    print("delay_of_y: ", delay_of_y)
                    print("P_bar = ", P_bar)
                    print("P_square_bar = ", P_square_bar)

                for r in range(0, M + 1):
                    if r == 0:
                        D_M_r[r] = P_bar
                        alpha_M_r[r] = P_bar
                        if FOSModel.DEBUG:
                            print("D[%d][%d] = %10.5f" % (M, r, D_M_r[r]))
                            print("alpha[%d][%d] = %10.5f" % (M, r, alpha_M_r[r]))
                    elif r > 0 and r < M:
                        D_M_r[r] = self.calculate_time_average_of_P_M_times_P_r(P_M, P[r], self.N_0, len(X)) - _np.sum([Alpha[r][i] * D_M_r[i] for i in range(0, r)])
                        alpha_M_r[r] = D_M_r[r] / D[r][r]
                        if FOSModel.DEBUG:
                            print("D[%d][%d] = %10.5f" % (M, r, D_M_r[r]))
                            print("alpha[%d][%d] = %10.5f" % (M, r, alpha_M_r[r]))
                    elif r == M:
                        D_M_r[r] = self.calculate_time_average_of_P_square(P_M, self.N_0, len(X)) - _np.sum([alpha_M_r[i] * D_M_r[i] for i in range(0, r)])
                        if FOSModel.DEBUG:
                            print("D[%d][%d] = %10.5f" % (M, r, D_M_r[r]))
                    else:
                        raise ValueError("wrong r value!")

                C_M = self.calculate_time_average_of_P_M_times_y(P_M, Y, self.N_0, len(X)) - _np.sum([alpha_M_r[r] * C[r] for r in range(0, M)])
                if FOSModel.DEBUG:
                    print("C(%d) for the %dth candidata = %10.5f" % (M, index, C_M))

                g_M = C_M / D_M_r[M]
                Q_M = g_M ** 2 * D_M_r[M]

                if FOSModel.DEBUG:
                    print("Q(%d) for the %dth candidata = %10.5f" % (M, index, Q_M))

                if Q_M > MAX_Q_M:
                    MAX_Q_M = Q_M
                    INDEX_OF_BEST_P_M = index
                    P[M] = P_M
                    P_in_polynomial_form[M] = self.candidate_list[index]
                    C[M] = C_M
                    D[M][M] = D_M_r[M]
                    g[M] = g_M
                    for r in range(0, M):
                        Alpha[M][r] = alpha_M_r[r]

            actual_len_of_P = M + 1

            if FOSModel.DEBUG:
                print("\n################### finished choosing candidate for M=%d, best Q_M is: %10.5f, corresponding index is: %d" % (M, MAX_Q_M, INDEX_OF_BEST_P_M), "###################")

            should_stop_early = (MAX_Q_M <= self.MSE_REDUCTION_THRESHOLD)
            if should_stop_early:
                if FOSModel.DEBUG:
                    print("\nthe last term contributes very little to the model, it will be removed")
                actual_len_of_P = actual_len_of_P - 1  # remove the last term since it contribute very little to the model

            MSE = _np.mean(Y[self.N_0:] ** 2) - _np.sum([ g[m] ** 2 * D[m][m] for m in range(0, actual_len_of_P) ])

            # calculate coefficient
            a_m = [0] * (actual_len_of_P)
            v = [0] * (actual_len_of_P)
            for m in range(0, actual_len_of_P):
                v[m] = 1
                for i in range(m + 1, actual_len_of_P):
                    v[i] = -1 * _np.sum([ Alpha[i][r] * v[r] for r in range(m, actual_len_of_P - 1) ])
                a_m[m] = _np.sum([_np.asarray(g[i]) * v[i] for i in range(m, actual_len_of_P)])

            if should_stop_early or M >= self.MAX_M or MSE <= self.MSE_THRESHOLD:
                if FOSModel.DEBUG:
                    print("\nstop adding terms.MSE: ", MSE)
                    print("by MSE threshold reduction? : ", should_stop_early)
                    print("by MSE threshold? : ", MSE < self.MSE_THRESHOLD)
                self.final_P = P[:actual_len_of_P]
                self.final_P_in_polynomial_form = P_in_polynomial_form[:actual_len_of_P]
                self.final_a_m = a_m
                print("end fitting, time used: ", time.time() - start_time)
                return (self.final_P_in_polynomial_form, self.final_a_m)

            # got to add next term
            self.candidate_list.pop(INDEX_OF_BEST_P_M)
            M += 1

    def predict(self, X, Y):

        if len(self.final_P_in_polynomial_form) == 0:
            raise Exception("model is empty, fit first!")

        if(len(Y) > len(X)):
            raise ValueError("output is longer than input!")

        Y_predicted = _np.array([])

        for n in range(len(Y), len(X)):
            y = 0
            for i in range(0, len(self.final_P_in_polynomial_form)):
                if i == 0:
                    y += self.final_a_m[i] * 1  # constant term
                else:
                    delay_of_x = self.final_P_in_polynomial_form[i][0]
                    delay_of_y = self.final_P_in_polynomial_form[i][1]
                    if delay_of_x == None and delay_of_y == None:
                        raise ValueError('this cannot be a constant term')
                    p = 1
                    if delay_of_x != None:
                        for l in delay_of_x:
                            p = p * (X[n - l] if n >= l else 0)
                    if delay_of_y != None:
                        for k in delay_of_y:
                            p = p * (Y[n - k] if n >= k else 0)
                    y += self.final_a_m[i] * p
            Y_predicted = _np.append(Y_predicted, y)
            Y = _np.append(Y, y)
        return Y, Y_predicted

    def get_printable_function(self):
        f = "y = " + str(self.final_a_m[0])
        for i in range(1, len(self.final_P_in_polynomial_form)):
            delay_of_x = self.final_P_in_polynomial_form[i][0]
            delay_of_y = self.final_P_in_polynomial_form[i][1]
            if delay_of_x == None and delay_of_y == None:
                        raise ValueError('this cannot be a constant term')
            current_term = str(self.final_a_m[i])
            if delay_of_x != None:
                for l in delay_of_x:
                    current_term = current_term + "*x[n-" + str(l) + "]"
            if delay_of_y != None:
                for k in delay_of_y:
                    current_term = current_term + "*y[n-" + str(k) + "]"
            f = f + " + " + current_term
        return f

    # end def get_printable_function
# end class FOSModel

#  ============================================================== #

def combsrep(v, k):
    """
    COMBSREP Combinations with replacement.

     COMBSREP(V, K) where V is a vector of length N, produces a matrix with
     (N+1-1)!/K!(N-1)! (i.e., "N+K-1 choose K") rows and K columns. Each row of
     the result has K of the elements in the vector V. The vector V may be a
     vector of any class. If the input is sparse, the output will be sparse.

     COMBSREP(V, K) lists all possible ways to pick K elements out of the vector
     V, with replacement, but where the order of the K elements is irrelevant.

     Author: Peter J. Acklam
     Time-stamp: 2003-08-22 08:00:51 +0200
     E-mail: pjacklam@online.no
     URL: http://home.online.no/~pjacklam

     converted to python by GMW (untested, quick and dirty): 2021
    """
    # error(nargchk(2, 2, nargin));
    v = _np.atleast_1d(_np.asarray(v))
    v = v.squeeze()
    if len(_np.shape(v)) > 1:
        print('First argument must be a vector.')
        return None
    # end if

    k = _np.atleast_1d(_np.asarray(k))
    k = v.squeeze()
    if  (len(k) != 1) or (k != _np.round(k)) or (k < 0):
        print('Second argument must be an integer scalar.');
        return None
    # end if

    if k == 0:
        # y = _np.zeros((0, k))
        y = _np.zeros((k,), dtype=v.dtype)
    elif k == 1:
        # y = _np.zeros((len(v), 1))
        # for ii in range(len(v)): # i = 1:length(v)
        #     y[ii] = _np.copy(v[ii])
        # # end for
        # y = _np.zeros((len(v),), dtype=v.dtype)
        y = _np.copy(v)
    else:
        v = _np.flatten(v) # v = v[:]
        y = []
        m = len(v)
        if m == 1:
            # y = v[0, _np.ones((k, 0))]
            y = _np.copy(v[:k])
        else:
            for ii in range(m):  # i = 1 : m
                y_recr = combsrep(v[ii:], k-1)  # combsrep(v(i:end), k-1);
                s_repl = v[ii]
                s_repl = s_repl[_np.ones((_np.size(y_recr, axis=0), 1)), 1]
                y = _np.vstack((y, _np.hstack((s_repl, y_recr)))) # [ y ; s_repl, y_recr ];
            # end for
        # endif
    # endif

    return y

def delay( x, k ):
    # DELAY Summary of this function goes here
    #   Detailed explanation goes here
    return _np.vstack((_np.zeros((k,_np.size(x,axis=1))); x[:-k]))

def evalfunct( x, y, p, a ):
    # EVALFUNCT Summary of this function goes here
    #   Detailed explanation goes here
    y1 = _np.ones( (len(x),1) )
    y1 *= a[0]
    for jj in range(1,len(p)): # j=2:length(p)
        y1 += a[jj] * evalterm(x, y, p[jj])
    # end for
    return y1

def evalterm( x, y, lags)
    # GETFUNCTION Summary of this function goes here
    #   Detailed explanation goes here

    p = _np.ones((len(x),1))
    for ii in range(len(lags.x)): # i=1:length(lags.x)
        k = lags.x[ii]
        p *= delay(x,k)
    # end for

    for ii in range(len(lags.y)): # i=1:length(lags.y)
        l = lags.y[ii]
        p *= delay(y,l)
    # end for
    return p

def lnl(g, a, k, x):
    # [y, u] = lnl( g, a, k, x )
    # LNL Summary of this function goes here
    #   Detailed explanation goes here
    from scipy.signal import lfilter

    # u = filter(1, g, x)
    u = lfilter(1, g, x)

    v = _np.zeros(_np.shape(u))
    for ii in range(len(a)): # i=1:length(a)
        v += a[i]*u**ii
    # end for

    # y = filter(1, k, v);
    y = lfilter(1, k, v)
    return y, u

def printfunct(p, a):
    """
    function str = printfunct( p, a )
    PRINTFUNCT Prints the symbolic function in text format
       Given the coefficients a, and polynomials
       terms p, print and return the function in text format,
       e.g., "y1[n] = 0.5*x[n] + 0.2*x^2[n-1]*y{n]
    """
    # strout = ''
    # strout += 'yest[n] = ' + num2str(a1)
    strout = 'yest[n] = %3.2f'%(a1,)
    for jj in range(1, len(p)): # j=2:length(p)
        #strout += ' ' + num2str(a[jj]) + '*' + printterm(p[jj])
        strout += ' %3.2f*%s'%(a[jj], printterm(p[jj]))
    # end for
    print(strout)
    return strout

def printterm(lags):
    """
    function str = printterm( lags)

    PRINTTERM prints polynomial term
       Prints and returns string representation
       of a polynomial term as a product of
       lags
    """

    strout = ''

    for ii in range(len(lags.x)): # i=1:length(lags.x)
        k = lags.x[ii]
        if k == 0:
            strout += 'x[n]'
        else:
            # strout += 'x[n - ' + num2str(k) + ']'
            strout += 'x[n - %i]'%(k,)
        # endif

        if ii < len(lags.x):
            strout += '*'
        # endif
    # endfor

    for ii in range(len(lags.y)): # i=1:length(lags.y)
        l = lags.y[ii]
        if l == 0:  # this condition should never happen
            strout += 'y[n]'
        else:
            # strout += 'y[n - ' + num2str(l) + ']'
            strout += 'y[n - %i]'%(l,)
        # endif

        if ii < len(lags.y):
            strout += '*'
        # endif
    # endfor

    print(strout)
    return strout

def fos(x, y, K, L, order):
    """
    # https://github.com/mostafaelhoushi/FOS/blob/master/fos.m

    FOS Summary of this function goes here
       Detailed explanation goes here
    """
    N = len(x)
    N0 = _np.max((K,L))

    # h = waitbar(0,'1','Name','FOS Calculation...',...
    #     'CreateCancelBtn',...
    #     'setappdata(gcbf,''canceling'',1)');
    # setappdata(h,'canceling',0)

    # ========== #

    # Structure of p:
    # p.x = delays of different x terms.
    # p.y = delays of different y terms.
    # p.x + p.y <= order
    #
    # p = struct('const', 1, 'x', [], 'y', []);
    # P = struct(p);
    from pybaseutils import Struct as struct
    p = {'const':1, 'x':[], 'y':[]}
    P = struct(p)

    # ========== #

    g, C, P, Q = [[] for _ in range(4)]
    D = [[1],[1]]

    g[0] = _np.mean( y[N0+1:N] )
    D[0,0] = 1
    C[0] = _np.mean( y[N0+1:N] )
    P[0] = []
    Q[0] = g[0]**2.0 * D[0,0]

    # waitbar(0, h,'Generating Candidates...');

    #  generate all candidates
    ii = 1
    for torder in range(order): # torder = 1 : order
        # waitbar(torder / order, h);
        for xorder in range(torder): # xorder = 0:torder
            # if getappdata(h,'canceling')
            #     delete(h);
            #     return;
            # end if

            yorder = torder - xorder

            xdelays = combsrep(_np.asarray(range(0, K)), xorder)
            ydelays = combsrep(_np.asarray(range(1, L)), yorder)

            if (_np.size(xdelays,axis=0) >= 1):
                for jj in range(_np.size(xdelays, axis=0): # j = 1:size(xdelays,1)
                    P[ii].x = xdelays[jj, :]
                    if _np.size(ydelays,axis=0) >= 1:
                        for kk in range(ydelays, axis=0): # k = 1:size(ydelays,1)
                            P[ii].y = ydelays[kk, :]
                            ii += 1
                        # end for
                    else:
                        ii += 1
                    # end if
                # end for
            else:
                for kk in range(_np.size(ydelays, axis=0)): # k = 1:size(ydelays,1)
                    P[ii].y = ydelays[kk, :]

                    ii += 1;
                # end for
            # end if
        # end for
    # end for
    #
    # waitbar(0, h, 'Evaluating Candidates...');

    M = 1
    while True and M<2e3:
        # if getappdata(h,'canceling')
        #     delete(h);
        #     return;
        # end if
        #
        # waitbar(0, h, sprintf('Evaluating Candidate %d...', M));
        mm = _np.copy(M)

        # Evaluate Q for each candidate
        try:            del Qc
        except:         pass
        # clear Qc;
        if P is None: # (isempty(P))
            break
        # end if
        for ii in range(len(P)): # i=1:length(P)
            # if getappdata(h,'canceling')
            #     delete(h);
            #     return;
            # end if
            #
            # waitbar(i / length(P), h);
            Pval = evalterm(x, y, P[ii])

            # D(m+1,1) = mean(Pval(N0+1:N));
            D[mm,0] = _np.mean(Pval[N0+1:N])
            for jj in range(mm): # j=1:m
                # if getappdata(h,'canceling')
                #     delete(h);
                #     return;
                # end if

                # alpha(m+1, j) = D(m+1, j) ./ D(j, j);
                alpha[mm, jj] = D[mm, jj] / D[jj, j]
                if (jj < M):
                    pval = evalterm(x, y, p[jj])
                else
                    pval = _np.copy(Pval)
                # end if
                # D(m+1, j+1) = mean(Pval(N0+1:N) .* pval(N0+1:N)) - sum(alpha(j+1, 1:j) .* D(m+1, 1:j));
                D[mm, jj] = _np.mean(Pval[N0+range(N)] * pval[N0+range(N)]) - _np.sum(alpha[jj, :jj] * D[mm, :jj])
            end
            C(m+1) = mean(y(N0+1:N) .* Pval(N0+1:N)) - sum(alpha(m+1, 1:m) .* C(1:m));

            g(m+1) = C(m+1)/D(m+1, m+1);
            Qc(i) = g(m+1)^2 * D(m+1, m+1);
        end

        % Find index of maximum Q
        index = find(Qc == max(Qc));
        Pval = evalterm(x, y, P(index(1)));

        for j=1:m
            if getappdata(h,'canceling')
                delete(h);
                return;
            end

            D(m+1,1) = mean(Pval(N0+1:N));
            for j=1:m
                if getappdata(h,'canceling')
                    delete(h);
                    return;
                end

                alpha(m+1, j) = D(m+1, j) ./ D(j, j);
                if (j < M)
                    pval = evalterm(x, y, p(j+1));
                else
                    pval = Pval;
                end
                D(m+1, j+1) = mean(Pval(N0+1:N) .* pval(N0+1:N)) - sum(alpha(j+1, 1:j) .* D(m+1, 1:j));
            end

        end
        # C(m+1) = mean(y(N0+1:N) .* Pval(N0+1:N)) - sum(alpha(m+1, 1:m) .* C(1:m));
        C[mm+1] = _np.mean(y[N0+1:N] * Pval[N0+1:N]) - _np.sum(alpha[mm, :mm] * C[:m])
        Q[mm+1] = _np.max(Qc)

        diagD = _np.diag(D).T;   # diag(D)'  <- conjugate transpose
        if ( Q[M+1] < 4/(N - N0 + 1) *(_np.mean(y[N0+1:N]**2.0) - _np.sum(Q[1:M]))):
            M -= 1
            break
        # end if

        # p(m+1) = P(index(1));
        # P(index(1)) = []; % remove it from the P
        p[mm] = P[index[0]]
        P[index[0]] = [] # remove it from the P

        # find the coefficient of the chosen candidate
        # g(m+1) = C(m+1) / D(m+1,m+1);
        g[mm] = C[mm] / D[mm,mm]

        if (P is None or Q[mm]<1e-26 ):  # if (isempty(P) || Q(m+1)<1e-26 )
            break
        # end if

        M += 1
    # end

    if plotit:
        _plt.figure(2)
        _plt.plot(_np.asarray(range(len(Q))), Q)
        _plt.ylabel('Q[m]')
        _plt.xlabel('m')
    # end if

    # Obtain a
    for ii in range(M): # i=0:M
        v[ii +1] = 1
        for jj in range(ii+1, M): # j = i+1:M
            v[jj +1] = -_np.sum(alpha[jj +1, ii +1 : jj-1 +1] * v[ii +1 : jj-1 +1])
        # end for
        a[ii +1] = _np.sum( g[ii +1:M +1]*v[ii +1:M +1])
    # end for


    return a, p, Q


#  ============================================================== #
#  ============================================================== #

def test1():
    ## Initialize
    # set(0,'DefaultFigureWindowStyle','docked');
    from numpy.random import default_rng
    import matplotlib.pyplot as _plt

    P = 0
    N = 1000

    K = 6
    L = 5
    order = 2

    N0 = _np.max((K,L))
    ## Generate x
    # Uniformly distributed pseudorandom numbers [0,1]
    rng = default_rng(0)
    x = rng.rand(N, 1)

    ## Generate y
    y = _np.zeros((N, 1))
    for nn in range(2, N): # n = 3:N
        # y(n) = sin(x(n-1))*cos(x(n)) + exp(-3*x(n))*sqrt(abs(x(n))) + 0.1*log(abs(y(n-2)+0.01))*y(n-1);
        y[nn] = (_np.sin(x[nn-1])*_np.cos(x[nn])
               + _np.exp(-3*x[nn])*_np.sqrt(_np.abs(x[nn]))
               + 0.1*_np.log(_np.abs(y[nn-2]+0.01))*y[nn-1]
               )
    # end for

    ## Apply FOS
    #tic
    [a, p] = fos( x[:N], y[:N], K, L, order )
    # toc
    y1 = evalfunct( x[:N], y[:N], p, a )
    e = y1 - y[:N]
    MSEpercent = 100 * _np.mean(e[N0+1:N]**2.0)/_np.var(y[N0+1:N])

    ## Print Results
    print('FOS generated model: ')
    printfunct(p, a)
    print('')
    print('MSE/%= %3.2f'%(MSEpercent,))

    ## Plot Results
    _plt.figure()
    _plt.plot(y, 'b')
    #hold on;
    _plt.plot(y1, 'r')
    #legend('y', 'y1')
    _plt.xlabel('n')
    _plt.ylabel('y[n]')
# end test1

def test2():
    from numpy.random import default_rng
    import matplotlib.pyplot as _plt

    P = 0
    N = 1000

    K = 3
    L = 3
    order = 3

    N0 = _np.max((K,L))
    ## Generate x
    # Uniformly distributed pseudorandom numbers [0,1]
    rng = default_rng(1)
    x1 = rng.rand(N, 1)
    rng = default_rng(2)
    x2 = rng.rand(N, 1);

    ## Generate y
    y1 = _np.zeros((N, 1))
    y2 = _np.zeros((N, 1))
    for nn in range(2,N): # n = 3:N
        y1[nn] = (
            0.75 + 0.9*x1[nn-2]*x2[nn-1]*x1[nn] + 3.2*x1[nn] + 2.8*x2[nn-2]**2.0 + 0.8*x2[nn-1]*y2[nn-1]
            )
        y2[nn] = (
            0.9 + 0.3*x2[nn-1]*x1[nn-2]*y1[nn-1] + 5*x2[nn-1] + 1.4*x1[nn-1]**2.0 + 0.1*x1[nn-2]*y2[nn-2]
            )
    # end for

    ## Apply FOS
    # tic
    [a, p, Q] = fos( _np.hstack((x1, x2)), _np.hstack((y1, y2)), K, L, order )
    # toc

    _plt.figure() #  hold on;
    _plt.plot(_np.asarray(range(len(a{1}))), Q[0, :len(a{1})], 'g')
    _plt.plot(_np.asarray(range(len(a{2}))), Q[1, :len(a{2})], 'y')
    _plt.ylabel('Q[m]')
    _plt.xlabel('m')
    #legend('y1', 'y2')

    yest = evalfunct( _np.hstack((x1, x2)), _np.hstack((y1, y2)), p, a )
    y1est = yest[:,0]
    y2est = yest[:,1]
    e1 = y1est - y1
    MSEpercent1 = _np.mean(e1[N0+1:N]**2)/_np.var(y1[N0+1:N]) *100
    e2 = y2est - y2
    MSEpercent2 = _np.mean(e2[N0+1:N]**2)/_np.var(y2[N0+1:N]) *100

    ## Print Results
    print('FOS generated model: ')
    printfunct(p, a)
    print('')
    print('MSE/%= %3.2f'%(MSEpercent,))

    ## Plot Results
    _plt.figure()
    _plt.plot(y1, 'b') # hold on
    _plt.plot(y1est, 'r')
    # legend('y1', 'y1est')
    _plt.xlabel('n')
    _plt.ylabel('y[n]')
    _plt.figure()
    _plt.plot(y2, 'b') # hold on
    _plt.plot(y2est, 'r')
    # legend('y2', 'y2est')
    _plt.xlabel('n')
    _plt.ylabel('y[n]')
# end def test2

def test_project():
    """
    https://github.com/mostafaelhoushi/FOS/blob/master/project.m
    """
    NotImplementedError
    """
    ## Initialize
    set(0,'DefaultFigureWindowStyle','docked');
    close all;
    clear all;
    clc;
    P = 0;
    N = 1000;

    ## Generate x
    # Uniformly distributed pseudorandom numbers [0,1]
    # rng(0);
    # x = rand(3*N, 1);

    # Uniform distribution
    # sigmax = 1;
    # rng(0);
    # x1 = rand(3*N, 1);
    # x = x1 / std(x1) * sigmax;

    # Normal distribution
    # sigmax = 1;
    # rng(0);
    # x1 = randn(3*N, 1);
    # x = x1 / std(x1) * sigmax;

    # Sinusoidal
    f = 1;
    A = 1;
    t = (1:3*N)';
    x = A*sin(2*pi*f*t);

    # Triangular
    # width = 1;
    # A = 1;
    # t = (1:3*N)';
    # x = A*sawtooth(t,width);

    ## Generate y
    # Simple Difference Equation 1
    y = 1 +  0.6*x  + 0.3*delay(x,1) + 0.4*delay(x,2) + 0.7*delay(x,3);

    # Simple Difference Equation 2
    # y = zeros(3*N, 1);
    # for n = 2:3*N
    #     y(n) = 2 + 0.5*x(n)*y(n-1);
    # end

    # Complex Difference Equation
    # y = zeros(3*N, 1);
    #
    # a0 = 1;
    # b0 = 0.7;
    # b1 = 0.8;
    # c1 = 0.1;
    # c2 = 0.4;
    # c4 = 0.2;
    #
    # for n = 6:3*N
    #     y(n) = a0 + b0*x(n) + b1*x(n-1) + c1*x(n-1)*y(n-1) - c2*x(n)*x(n-1)+ c4*x(n-4)*x(n-5);
    # end

    # LNL Cascade
    # i = 1:5;
    # g1 = exp(-i) + exp(-2*i);
    # a1 = 0.5 + 2*exp(-i);
    # k1 = 3*exp(-i);
    #
    # g2 = exp(-i) + 3*exp(-2*i);
    # a2 = 0.2 + 3*exp(-i);
    # k2 = 0.1*exp(-i) + 0.9*exp(-2*i);
    #
    # y = lnl(g1, a1, k1, x) + lnl(g2, a2, k2, x);

    # Non-Polynomial Equation
    # y = zeros(3*N, 1);
    # for n = 3:3*N
    #     y(n) = sin(x(n-1))*cos(x(n)) + exp(-3*x(n))*sqrt(abs(x(n))) + 0.1*log(abs(y(n-2)+0.01))*y(n-1);
    # end

    # Real Data 1 - Project 2
    # load wz.mat;
    # wzd = wden(wz, 'heursure', 's', 'one', 15, 'db4');
    # x = (1:3*N)';
    # y = wzd(1:3*N);
    # w = wden(wz, 'heursure', 's', 'one', 14, 'db4'); w=w(1:length(x));
    # r = w - y;
    # P = 100 * var(r)/var(y);

    # Real Data 2 - Project 3
    # load 'C:\Data Logging\EE 517 Winter 2012\Project 3\Xbow_stat_data_other.mat';
    # x = (1:3*N)';
    # y = f.x(1:3*N);
    # clear f w interp_info denoising_info denoising_info_ orig_data_info;

    # Noise Insertion
    # rng(1);
    # r1 = wgn(3*N, 1, 0);
    # r1 = r1 - mean(r1);
    # r = r1 / std(r1) * sqrt(P/100) * std(y);
    # w = r + y;

    # K = 6;
    # L = 5;
    # order = 5;
    # N0 = max(K,L);
    # tic
    # [a, p] = fos( x(1:N), y(1:N), K, L, order );
    # toc
    # y1 = evalfunct( x(1:N), y(1:N), p, a );
    # e = y1 - y(1:N);
    # MSEpercent = mean(e(N0+1:N).^2)/var(y(N0+1:N)) *100;


    ## Training Phase
    tic
    # Apply FOS for 1st 1000 samples of data
    rng(2);
    K = randi([3,20], 10, 1);
    rng(3);
    L = randi([3,20], 10, 1);
    for i = 1:length(K)
        order = 2;
        N0 = max(K(i),L(i));
        #tic
        [at{i}, pt{i}] = fos( x(1:N), w(1:N), K(i), L(i), order );
        #toc

        y1 = evalfunct( x(1:N), w(1:N), pt{i}, at{i} );
        e = y1 - w(1:N);
        MSEpercent(i) = mean(e(N0+1:N).^2)/var(w(N0+1:N)) *100;
    end

    ## Selection Phase
    clear MSEpercent;
    # Apply FOS for 2nd 1000 samples of data
    for i = 1:1:length(K)
        y1 = evalfunct( x(N+1:2*N), w(N+1:2*N), pt{i}, at{i} );
        e = y1 - w(N+1:2*N);
        MSEpercent(i) = mean(e(N0+1:N).^2)/var(w(N+N0+1:2*N)) *100;
    end

    # Choose the best model over the 2nd 1000 samples
    index = find(MSEpercent == min(MSEpercent));
    as = at{index(1)};
    ps = pt{index(1)};


    ## Evaluation Phase
    clear MSEpercent;
    # Apply FOS for 3nd 1000 samples of data
    y1 = evalfunct( x(2*N+1:3*N), w(2*N+1:3*N), ps, as);
    e = y1 - y(2*N+1:3*N);
    MSEpercent = mean(e(N0+1:N).^2)/var(y(2*N+N0+1:3*N)) *100;
    IdealMSEPercent = var(r) / var(w) * 100;

    e = y1 - w(2*N+1:3*N);
    MSEpercent1 = mean(e(N0+1:N).^2)/var(w(2*N+N0+1:3*N)) *100;


    ## Plot
    y1 = evalfunct( x, w, ps, as);
    figure(1);
    plot(y, 'b'); hold on;
    plot(N0+1:3*N, y1(N0+1:3*N),'r');
    xlabel('n');
    ylabel('y[n]');
    legend('y', 'y1');

    figure(3);
    Rw = xcorr(w - mean(w));
    plot(-length(Rw)/2:length(Rw)/2-1 , Rw);
    title('Auto-correlation of output w[n]');
    xlabel('n');
    ylabel('R(w)');

    toc

    ## Plot Q[m] for best K & L
    order = 2;
    N0 = max( K(index(1)) , L(index(1)) );
    [a, p] = fos( x(1:N), y(1:N), K(index(1)), L(index(1)), order );
    """
# end def test_project

#  ============================================================== #

def test_FOSmodel1():
    mu, sigma = 0, 3.5 # mean and standard deviation
    x = _np.random.normal(mu, sigma, 1000)
    y = _np.sin(x)

    # where max_m is the max number of  terms, mse_reduction_threshold is the early-stop criteria
    model = FOSModel(max_delay_in_input=0, max_delay_in_output=0, max_order=20, max_m=10, mse_reduction_threshold=1e-4)

    # x (input) and y (output) are 1-dimentional series
    model.fit(x, y)

    # all x points and a few initial points of y, predicted y series will be returned
    _, y1 = model.predict(x, _np.zeros(0))

    mse = _np.mean((y1 - y)**2)

    print("y: ", y.shape)
    print("y1: ", y1.shape)

    print("mse: ", mse)

if __name__=='__main__':
    test_FOSmodel1()

# end if





#  ============================================================== #
#  ============================================================== #