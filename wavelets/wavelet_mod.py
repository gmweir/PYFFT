# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:52:16 2020

@author: gawe


if we use pywavelets in a scientific publication cite:
    Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019).
    PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237,
    https://doi.org/10.21105/joss.01237.

    version 1.0.3 comes with my distribution of anaconda in py3.7

if we use pycwt in a scientific publication cite:
    ... Unnecessary to cite the code itself as it is a port of other codes,
        but the method is based on the Torrence and Compo (1998) method:
    Torrence, C. and Compo, G.P.. A practical guide to Wavelet Analysis.
    Bulletin of the American Meteorological Society, American Meteorological
    Society, 1998, 79, 61-78.

"""
# ========================================================================= #
# ========================================================================= #

import numpy as np
#import os as os
import matplotlib.pyplot as plt

#try:
#    import FIT.pycwt
#except:
#    from . import pycwt
## end try


# ========================================================================= #
# ========================================================================= #


class demo(object):
    @staticmethod
    def batch_processing():
        """
        Demo: Parallel processing accross images
        Multithreading can be used to run transforms on a set of images in parallel.
        This will give a net performance benefit if the images to be transformed are
        sufficiently large.
        This demo runs a multilevel wavelet decomposition on a list of 32 images,
        each of size (512, 512).  Computations are repeated sequentially and in
        parallel and the runtimes compared.
        In general, multithreading will be more beneficial for larger images and for
        wavelets with a larger filter size.
        One can also change ``ndim`` to 3 in the code below to use a set of 3D volumes
        instead.
        """
        import pywt
        import time
        from functools import partial
        from multiprocessing import cpu_count

        try:
            from concurrent import futures
        except ImportError:
            raise ImportError(
                "This demo requires concurrent.futures.  It can be installed for "
                "for python 2.x via:  pip install futures")

        import numpy as np
        from numpy.testing import assert_array_equal


        # the test image
        cam = pywt.data.camera().astype(float)

        ndim = 2                   # dimension of images to transform (2 or 3)
        num_images = 32            # number of images to transform
        max_workers = cpu_count()  # max number of available threads
        nrepeat = 5                # averages used in the benchmark

        # create a list of num_images images
        if ndim == 2:
            imgs = [cam, ] * num_images
            wavelet = 'db8'
        elif ndim == 3:
            # stack image along 3rd dimension to create a [512 x 512 x 16] 3D volume
            im3 = np.concatenate([cam[:, :, np.newaxis], ]*16, axis=-1)
            # create multiple copies of the volume
            imgs = [im3, ] * num_images
            wavelet = 'db1'
        else:
            ValueError("Only 2D and 3D test cases implemented")

        # define a function to apply to each image
        wavedecn_func = partial(pywt.wavedecn, wavelet=wavelet, mode='periodization',
                                level=3)


        def concurrent_transforms(func, imgs, max_workers=None):
            """Call func on each img in imgs using a ThreadPoolExecutor."""
            executor = futures.ThreadPoolExecutor
            if max_workers is None:
                # default to as many workers as available cpus
                max_workers = cpu_count()
            results = []
            with executor(max_workers=max_workers) as execute:
                for result in execute.map(func, imgs):
                    results.append(result)
            return results


        print("Processing {} images of shape {}".format(len(imgs), imgs[0].shape))

        # Sequential computation via a list comprehension
        tstart = time.time()
        for n in range(nrepeat):
            results = [wavedecn_func(img) for img in imgs]
        t = (time.time()-tstart)/nrepeat
        print("\nSequential Case")
        print("\tElapsed time: {:0.2f} ms".format(1000*t))


        # Concurrent computation via concurrent.futures
        tstart = time.time()
        for n in range(nrepeat):
            results_concurrent = concurrent_transforms(wavedecn_func, imgs,
                                                       max_workers=max_workers)
        t2 = (time.time()-tstart)/nrepeat
        print("\nMultithreaded Case")
        print("\tNumber of concurrent workers: {}".format(max_workers))
        print("\tElapsed time: {:0.2f} ms".format(1000*t2))
        print("\nRelative speedup with concurrent = {}".format(t/t2))

        # check a couple of the coefficient arrays to verify matching results for
        # sequential and multithreaded computation
        assert_array_equal(results[-1][0],
                           results_concurrent[-1][0])
        assert_array_equal(results[-1][1]['d' + 'a'*(ndim-1)],
                           results_concurrent[-1][1]['d' + 'a'*(ndim-1)])

    # end def

    @staticmethod
    def benchmark():
        import pywt
        import gc
        import sys
        import time

        if sys.platform == 'win32':
            clock = time.clock
        else:
            clock = time.time

        sizes = [20, 50, 100, 120, 150, 200, 250, 300, 400, 500, 600, 750,
                 1000, 2000, 3000, 4000, 5000, 6000, 7500,
                 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000,
                 100000, 150000, 200000, 250000, 300000, 400000, 500000,
                 600000, 750000, 1000000, 2000000, 5000000][:-4]

        wavelet_names = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7',
                         'db8', 'db9', 'db10', 'sym10', 'coif1', 'coif2',
                         'coif3', 'coif4', 'coif5']


        wavelets = [pywt.Wavelet(n) for n in wavelet_names]
        mode = pywt.Modes.zero

        times_dwt = [[] for i in range(len(wavelets))]
        times_idwt = [[] for i in range(len(wavelets))]

        for j, size in enumerate(sizes):
            data = np.ones((size,), dtype=np.float64)
            print((("%d/%d" % (j + 1, len(sizes))).rjust(6), str(size).rjust(9)))
            for i, w in enumerate(wavelets):
                min_t1, min_t2 = 9999., 9999.
                for _ in range(5):
                    # Repeat timing 5 times to reduce run-to-run variation
                    t1 = clock()
                    (a, d) = pywt.dwt(data, w, mode)
                    t1 = clock() - t1
                    min_t1 = min(t1, min_t1)

                    t2 = clock()
                    a0 = pywt.idwt(a, d, w, mode)
                    t2 = clock() - t2
                    min_t2 = min(t2, min_t2)

                times_dwt[i].append(min_t1)
                times_idwt[i].append(min_t2)

            gc.collect()


        for j, (times, name) in enumerate([(times_dwt, 'dwt'), (times_idwt, 'idwt')]):
            fig = plt.figure(j)
            ax = fig.add_subplot(111)
            ax.set_title(name)

            for i, n in enumerate(wavelet_names):
                ax.loglog(sizes, times[i], label=n)

            ax.legend(loc='best')
            ax.set_xlabel('len(x)')
            ax.set_ylabel('time [s]')


        plt.show()
    # end def demo

    @staticmethod
    def cwt_analysis():
        import pywt

        time, sst = pywt.data.nino()
        dt = time[1] - time[0]

        # Taken from http://nicolasfauchereau.github.io/climatecode/posts/wavelet-analysis-in-python/
        wavelet = 'cmor1.5-1.0'
        scales = np.arange(1, 128)

        [cfs, frequencies] = pywt.cwt(sst, scales, wavelet, dt)
        power = (abs(cfs)) ** 2

        period = 1. / frequencies
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        f, ax = plt.subplots(figsize=(15, 10))
        ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
                    extend='both')

        ax.set_title('%s Wavelet Power Spectrum (%s)' % ('Nino1+2', wavelet))
        ax.set_ylabel('Period (years)')
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(Yticks))
        ax.set_yticklabels(Yticks)
        ax.invert_yaxis()
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], -1)

        plt.show()
    # end def demo

    @staticmethod
    def dwt2_dwtn_image():
        import pywt
        import pywt.data


        # Load image
        original = pywt.data.camera()

        # Wavelet transform of image, and plot approximation and details
        titles = ['Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']
        coeffs2 = pywt.dwt2(original, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fig = plt.figure()
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("dwt2 coefficients", fontsize=14)

        # Now reconstruct and plot the original image
        reconstructed = pywt.idwt2(coeffs2, 'bior1.3')
        fig = plt.figure()
        plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

        # Check that reconstructed image is close to the original
        np.testing.assert_allclose(original, reconstructed, atol=1e-13, rtol=1e-13)


        # Now do the same with dwtn/idwtn, to show the difference in their signatures

        coeffsn = pywt.dwtn(original, 'bior1.3')
        fig = plt.figure()
        for i, key in enumerate(['aa', 'ad', 'da', 'dd']):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.imshow(coeffsn[key], interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("dwtn coefficients", fontsize=14)

        # Now reconstruct and plot the original image
        reconstructed = pywt.idwtn(coeffsn, 'bior1.3')
        fig = plt.figure()
        plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

        # Check that reconstructed image is close to the original
        np.testing.assert_allclose(original, reconstructed, atol=1e-13, rtol=1e-13)


        plt.show()
    # end def

    @staticmethod
    def dwt_signal_decomposition():
        import pywt
        import pywt.data

        ecg = pywt.data.ecg()

        data1 = np.concatenate((np.arange(1, 400),
                                np.arange(398, 600),
                                np.arange(601, 1024)))
        x = np.linspace(0.082, 2.128, num=1024)[::-1]
        data2 = np.sin(40 * np.log(x)) * np.sign((np.log(x)))

        mode = pywt.Modes.smooth


        def plot_signal_decomp(data, w, title):
            """Decompose and plot a signal S.

            S = An + Dn + Dn-1 + ... + D1
            """
            w = pywt.Wavelet(w)
            a = data
            ca = []
            cd = []
            for i in range(5):
                (a, d) = pywt.dwt(a, w, mode)
                ca.append(a)
                cd.append(d)

            rec_a = []
            rec_d = []

            for i, coeff in enumerate(ca):
                coeff_list = [coeff, None] + [None] * i
                rec_a.append(pywt.waverec(coeff_list, w))

            for i, coeff in enumerate(cd):
                coeff_list = [None, coeff] + [None] * i
                rec_d.append(pywt.waverec(coeff_list, w))

            fig = plt.figure()
            ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
            ax_main.set_title(title)
            ax_main.plot(data)
            ax_main.set_xlim(0, len(data) - 1)

            for i, y in enumerate(rec_a):
                ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
                ax.plot(y, 'r')
                ax.set_xlim(0, len(y) - 1)
                ax.set_ylabel("A%d" % (i + 1))

            for i, y in enumerate(rec_d):
                ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
                ax.plot(y, 'g')
                ax.set_xlim(0, len(y) - 1)
                ax.set_ylabel("D%d" % (i + 1))


        plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")
        plot_signal_decomp(data2, 'sym5',
                           "DWT: Frequency and phase change - Symmlets5")
        plot_signal_decomp(ecg, 'sym5', "DWT: Ecg sample - Symmlets5")


        plt.show()
    # end def

    @staticmethod
    def dwt_swt_show_coeffs():
        import pywt
        import pywt.data

        ecg = pywt.data.ecg()

        data1 = np.concatenate((np.arange(1, 400),
                                np.arange(398, 600),
                                np.arange(601, 1024)))
        x = np.linspace(0.082, 2.128, num=1024)[::-1]
        data2 = np.sin(40 * np.log(x)) * np.sign((np.log(x)))

        mode = pywt.Modes.sp1DWT = 1

        # Show DWT coefficients
        use_dwt = True
        demo.plot_coeffs(data1, 'db1', mode,
                    "DWT: Signal irregularity shown in D1 - Haar wavelet",
                    use_dwt)
        demo.plot_coeffs(data2, 'sym5', mode,
                    "DWT: Frequency and phase change - Symmlets5",
                    use_dwt)
        demo.plot_coeffs(ecg, 'sym5', mode,
                    "DWT: Ecg sample - Symmlets5", use_dwt)

        # Show DWT coefficients
        use_dwt = False
        demo.plot_coeffs(data1, 'db1', mode,
                    "SWT: Signal irregularity detection - Haar wavelet",
                    use_dwt)
        demo.plot_coeffs(data2, 'sym5', mode,
                    "SWT: Frequency and phase change - Symmlets5",
                    use_dwt)
        demo.plot_coeffs(ecg, 'sym5', mode,
                    "SWT: Ecg sample - simple QRS detection - Symmlets5",
                    use_dwt)


        plt.show()
    # end def

    @staticmethod
    def plot_coeffs(data, w, mode, title, use_dwt=True):
        """Show dwt or swt coefficients for given data and wavelet."""
        import pywt
        w = pywt.Wavelet(w)
        a = data
        ca = []
        cd = []

        if use_dwt:
            for i in range(5):
                (a, d) = pywt.dwt(a, w, mode)
                ca.append(a)
                cd.append(d)
        else:
            coeffs = pywt.swt(data, w, 5)  # [(cA5, cD5), ..., (cA1, cD1)]
            for a, d in reversed(coeffs):
                ca.append(a)
                cd.append(d)

        fig = plt.figure()
        ax_main = fig.add_subplot(len(ca) + 1, 1, 1)
        ax_main.set_title(title)
        ax_main.plot(data)
        ax_main.set_xlim(0, len(data) - 1)

        for i, x in enumerate(ca):
            ax = fig.add_subplot(len(ca) + 1, 2, 3 + i * 2)
            ax.plot(x, 'r')
            ax.set_ylabel("A%d" % (i + 1))
            if use_dwt:
                ax.set_xlim(0, len(x) - 1)
            else:
                ax.set_xlim(w.dec_len * i, len(x) - 1 - w.dec_len * i)

        for i, x in enumerate(cd):
            ax = fig.add_subplot(len(cd) + 1, 2, 4 + i * 2)
            ax.plot(x, 'g')
            ax.set_ylabel("D%d" % (i + 1))
            # Scale axes
            ax.set_xlim(0, len(x) - 1)
            if use_dwt:
                ax.set_ylim(min(0, 1.4 * min(x)), max(0, 1.4 * max(x)))
            else:
                vals = x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)]
                ax.set_ylim(min(0, 2 * min(vals)), max(0, 2 * max(vals)))
    # end def


    @staticmethod
    def fswavedecn():
        import pywt

        img = pywt.data.camera().astype(float)

        # Fully separable transform
        fswavedecn_result = pywt.fswavedecn(img, 'db2', 'periodization', levels=4)

        # Standard DWT
        coefs = pywt.wavedec2(img, 'db2', 'periodization', level=4)
        # convert DWT coefficients to a 2D array
        mallat_array, mallat_slices = pywt.coeffs_to_array(coefs)


        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(np.abs(mallat_array)**0.25,
                   cmap=plt.cm.gray,
                   interpolation='nearest')
        ax1.set_axis_off()
        ax1.set_title('Mallat decomposition\n(wavedec2)')

        ax2.imshow(np.abs(fswavedecn_result.coeffs)**0.25,
                   cmap=plt.cm.gray,
                   interpolation='nearest')
        ax2.set_axis_off()
        ax2.set_title('Fully separable decomposition\n(fswt)')

        plt.show()
    # end def

    @staticmethod
    def mondrian(shape=(256, 256), nx=5, ny=8, seed=4):
        """ Piecewise-constant image (reminiscent of Dutch painter Piet Mondrian's
        geometrical period).
        """
        rstate = np.random.RandomState(seed)
        min_dx = 0
        while(min_dx < 3):
            xp = np.sort(np.round(rstate.rand(nx-1)*shape[0]).astype(np.int))
            xp = np.concatenate(((0, ), xp, (shape[0], )))
            min_dx = np.min(np.diff(xp))
        min_dy = 0
        while(min_dy < 3):
            yp = np.sort(np.round(rstate.rand(ny-1)*shape[1]).astype(np.int))
            yp = np.concatenate(((0, ), yp, (shape[1], )))
            min_dy = np.min(np.diff(yp))
        img = np.zeros(shape)
        for ix, x in enumerate(xp[:-1]):
            for iy, y in enumerate(yp[:-1]):
                slices = [slice(x, xp[ix+1]), slice(y, yp[iy+1])]
                val = rstate.rand(1)[0]
                img[slices] = val
        return img

    @staticmethod
    def fswavedecn_mondrian():
        """Using the FSWT to process anistropic images.

        In this demo, an anisotropic piecewise-constant image is transformed by the
        standard DWT and the fully-separable DWT. The 'Haar' wavelet gives a sparse
        representation for such piecewise constant signals (detail coefficients are
        only non-zero near edges).

        For such anistropic signals, the number of non-zero coefficients will be lower
        for the fully separable DWT than for the isotropic one.

        This example is inspired by the following publication where it is proven that
        the FSWT gives a sparser representation than the DWT for this class of
        anistropic images:

        .. V Velisavljevic, B Beferull-Lozano, M Vetterli and PL Dragotti.
           Directionlets: Anisotropic Multidirectional Representation With
           Separable Filtering. IEEE Transactions on Image Processing, Vol. 15,
           No. 7, July 2006.

        """
        import pywt

        # create an anisotropic piecewise constant image
        img = demo.mondrian((128, 128))

        # perform DWT
        coeffs_dwt = pywt.wavedecn(img, wavelet='db1', level=None)

        # convert coefficient dictionary to a single array
        coeff_array_dwt, _ = pywt.coeffs_to_array(coeffs_dwt)

        # perform fully seperable DWT
        fswavedecn_result = pywt.fswavedecn(img, wavelet='db1')

        nnz_dwt = np.sum(coeff_array_dwt != 0)
        nnz_fswavedecn = np.sum(fswavedecn_result.coeffs != 0)

        print("Number of nonzero wavedecn coefficients = {}".format(np.sum(nnz_dwt)))
        print("Number of nonzero fswavedecn coefficients = {}".format(np.sum(nnz_fswavedecn)))

        img = demo.mondrian()
        fig, axes = plt.subplots(1, 3)
        imshow_kwargs = dict(cmap=plt.cm.gray, interpolation='nearest')
        axes[0].imshow(img, **imshow_kwargs)
        axes[0].set_title('Anisotropic Image')
        axes[1].imshow(coeff_array_dwt != 0, **imshow_kwargs)
        axes[1].set_title('Nonzero DWT\ncoefficients\n(N={})'.format(nnz_dwt))
        axes[2].imshow(fswavedecn_result.coeffs != 0, **imshow_kwargs)
        axes[2].set_title('Nonzero FSWT\ncoefficients\n(N={})'.format(nnz_fswavedecn))
        for ax in axes:
            ax.set_axis_off()

        plt.show()
    # end def


    @staticmethod
    def plot_demo_signals():
        """Plot the  set of 1D demo signals available in `pywt.data.demo_signal`."""
        import pywt

        # use 'list' to get a list of all available 1d demo signals
        signals = pywt.data.demo_signal('list')

        subplots_per_fig = 5
        signal_length = 1024
        i_fig = 0
        n_figures = int(np.ceil(len(signals)/subplots_per_fig))
        for i_fig in range(n_figures):
            # Select a subset of functions for the current plot
            func_subset = signals[
                i_fig * subplots_per_fig:(i_fig + 1) * subplots_per_fig]

            # create a figure to hold this subset of the functions
            fig, axes = plt.subplots(subplots_per_fig, 1)
            axes = axes.ravel()
            for n, signal in enumerate(func_subset):
                if signal in ['Gabor', 'sineoneoverx']:
                    # user cannot specify a length for these two
                    x = pywt.data.demo_signal(signal)
                else:
                    x = pywt.data.demo_signal(signal, signal_length)
                ax = axes[n]
                ax.plot(x.real)
                if signal == 'Gabor':
                    # The Gabor signal is complex-valued
                    ax.plot(x.imag)
                    ax.legend(['Gabor (Re)', 'Gabor (Im)'], loc='upper left')
                else:
                    ax.legend([signal, ], loc='upper left')
            # omit axes for any unused subplots
            for n in range(n + 1, len(axes)):
                axes[n].set_axis_off()
        plt.show()
    # end def demo

    @staticmethod
    def plot_wavelets():
        # Plot scaling and wavelet functions for db, sym, coif, bior and rbio families
        import pywt
        import itertools

        plot_data = [('db', (4, 3)),
                     ('sym', (4, 3)),
                     ('coif', (3, 2))]


        for family, (rows, cols) in plot_data:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=.02, left=.06,
                                right=.97, top=.94)
            colors = itertools.cycle('bgrcmyk')

            wnames = pywt.wavelist(family)
            i = iter(wnames)
            for col in range(cols):
                for row in range(rows):
                    try:
                        wavelet = pywt.Wavelet(next(i))
                    except StopIteration:
                        break
                    phi, psi, x = wavelet.wavefun(level=5)

                    color = next(colors)
                    ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols))
                    ax.set_title(wavelet.name + " phi")
                    ax.plot(x, phi, color)
                    ax.set_xlim(min(x), max(x))

                    ax = fig.add_subplot(rows, 2*cols, 1 + 2*(col + row*cols) + 1)
                    ax.set_title(wavelet.name + " psi")
                    ax.plot(x, psi, color)
                    ax.set_xlim(min(x), max(x))

        for family, (rows, cols) in [('bior', (4, 3)), ('rbio', (4, 3))]:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.5, wspace=0.2, bottom=.02, left=.06,
                                right=.97, top=.94)

            colors = itertools.cycle('bgrcmyk')
            wnames = pywt.wavelist(family)
            i = iter(wnames)
            for col in range(cols):
                for row in range(rows):
                    try:
                        wavelet = pywt.Wavelet(next(i))
                    except StopIteration:
                        break
                    phi, psi, phi_r, psi_r, x = wavelet.wavefun(level=5)
                    row *= 2

                    color = next(colors)
                    ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols))
                    ax.set_title(wavelet.name + " phi")
                    ax.plot(x, phi, color)
                    ax.set_xlim(min(x), max(x))

                    ax = fig.add_subplot(2*rows, 2*cols, 2*(1 + col + row*cols))
                    ax.set_title(wavelet.name + " psi")
                    ax.plot(x, psi, color)
                    ax.set_xlim(min(x), max(x))

                    row += 1
                    ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols))
                    ax.set_title(wavelet.name + " phi_r")
                    ax.plot(x, phi_r, color)
                    ax.set_xlim(min(x), max(x))

                    ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols) + 1)
                    ax.set_title(wavelet.name + " psi_r")
                    ax.plot(x, psi_r, color)
                    ax.set_xlim(min(x), max(x))

        plt.show()
    # end def

    @staticmethod
    def plot_wavelets_pyqtgraph():
        import pywt
        import sys
        from pyqtgraph.Qt import QtGui
        import pyqtgraph as pg


        families = ['db', 'sym', 'coif', 'bior', 'rbio']


        def main():
            app = QtGui.QApplication(sys.argv)
            tabs = QtGui.QTabWidget()

            for family in families:
                scroller = QtGui.QScrollArea()
                vb = pg.GraphicsWindow()
                vb.setMinimumHeight(3000)
                vb.setMinimumWidth(1900)
                scroller.setWidget(vb)
                for i, name in enumerate(pywt.wavelist(family)):
                    pen = pg.intColor(i)
                    wavelet = pywt.Wavelet(name)
                    if wavelet.orthogonal:
                        phi, psi, x = wavelet.wavefun(level=5)
                        ax = vb.addPlot(title=wavelet.name + " phi")
                        ax.plot(phi, pen=pen)
                        bx = vb.addPlot(title=wavelet.name + " psi")
                        bx.plot(psi, pen=pen)
                    else:
                        phi, psi, phi_r, psi_r, x = wavelet.wavefun(level=5)
                        ax = vb.addPlot(title=wavelet.name + " phi")
                        ax.plot(phi, pen=pen)
                        bx = vb.addPlot(title=wavelet.name + " psi")
                        bx.plot(psi, pen=pen)
                        ax = vb.addPlot(title=wavelet.name + " phi_r")
                        ax.plot(phi_r, pen=pen)
                        bx = vb.addPlot(title=wavelet.name + " psi_r")
                        bx.plot(psi_r, pen=pen)
                    if i % 2 == 0:
                        vb.nextRow()
                    # end if
                # end for
                tabs.addTab(scroller, family)
            # end for
            tabs.setWindowTitle('Wavelets')
            tabs.resize(1920, 1080)
            tabs.show()
            sys.exit(app.exec_())
        # end def
    # end def

    @staticmethod
    def swt2():
        import pywt
        import pywt.data


        arr = pywt.data.aero()

        plt.imshow(arr, interpolation="nearest", cmap=plt.cm.gray)

        level = 0
        titles = ['Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']
        for LL, (LH, HL, HH) in pywt.swt2(arr, 'bior1.3', level=3, start_level=0):
            fig = plt.figure()
            for i, a in enumerate([LL, LH, HL, HH]):
                ax = fig.add_subplot(2, 2, i + 1)
                # ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
                ax.imshow(a, origin='lower', interpolation="nearest", cmap=plt.cm.gray)
                ax.set_title(titles[i], fontsize=12)

            fig.suptitle("SWT2 coefficients, level %s" % level, fontsize=14)
            level += 1


        plt.show()
    # end def

    @staticmethod
    def swt_variance():
        import pywt
        import pywt.data

        ecg = pywt.data.ecg()

        # set trim_approx to avoid keeping approximation coefficients for all levels

        # set norm=True to rescale the wavelets so that the transform partitions the
        # variance of the input signal among the various coefficient arrays.

#        coeffs = pywt.swt(ecg, wavelet='sym4', trim_approx=True, norm=True)
        coeffs = pywt.swt(ecg, wavelet='sym4')

        ca = coeffs[0]
        details = coeffs[1:]

        print("Variance of the ecg signal = {}".format(np.var(ecg, ddof=1)))

        variances = [np.var(c, ddof=1) for c in coeffs]
        detail_variances = variances[1:]
        print("Sum of variance across all SWT coefficients = {}".format(
            np.sum(variances)))

        # Create a plot using the same y axis limits for all coefficient arrays to
        # illustrate the preservation of amplitude scale across levels when norm=True.
        ylim = [ecg.min(), ecg.max()]

        fig, axes = plt.subplots(len(coeffs) + 1)
        axes[0].set_title("normalized SWT decomposition")
        axes[0].plot(ecg)
        axes[0].set_ylabel('ECG Signal')
        axes[0].set_xlim(0, len(ecg) - 1)
        axes[0].set_ylim(ylim[0], ylim[1])

        for i, x in enumerate(coeffs):
            ax = axes[-i - 1]
            ax.plot(coeffs[i], 'g')
            if i == 0:
                ax.set_ylabel("A%d" % (len(coeffs) - 1))
            else:
                ax.set_ylabel("D%d" % (len(coeffs) - i))
            # Scale axes
            ax.set_xlim(0, len(ecg) - 1)
            ax.set_ylim(ylim[0], ylim[1])


        # reorder from first to last level of coefficients
        level = np.arange(1, len(detail_variances) + 1)

        # create a plot of the variance as a function of level
        plt.figure(figsize=(8, 6))
        fontdict = dict(fontsize=16, fontweight='bold')
        plt.plot(level, detail_variances[::-1], 'k.')
        plt.xlabel("Decomposition level", fontdict=fontdict)
        plt.ylabel("Variance", fontdict=fontdict)
        plt.title("Variances of detail coefficients", fontdict=fontdict)
        plt.show()
    # end def

    @staticmethod
    def wp_2d():
        import pywt
        from pywt import WaveletPacket2D
        import pywt.data


        arr = pywt.data.aero()

        wp2 = WaveletPacket2D(arr, 'db2', 'symmetric', maxlevel=2)

        # Show original figure
        plt.imshow(arr, interpolation="nearest", cmap=plt.cm.gray)

        path = ['d', 'v', 'h', 'a']

        # Show level 1 nodes
        fig = plt.figure()
        for i, p2 in enumerate(path):
            ax = fig.add_subplot(2, 2, i + 1)
            # ax.imshow(np.sqrt(np.abs(wp2[p2].data)), origin='image', interpolation="nearest", cmap=plt.cm.gray)
            ax.imshow(np.sqrt(np.abs(wp2[p2].data)), interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(p2)

        # Show level 2 nodes
        for p1 in path:
            fig = plt.figure()
            for i, p2 in enumerate(path):
                ax = fig.add_subplot(2, 2, i + 1)
                p1p2 = p1 + p2
                # ax.imshow(np.sqrt(np.abs(wp2[p1p2].data)), origin='image', interpolation="nearest", cmap=plt.cm.gray)
                ax.imshow(np.sqrt(np.abs(wp2[p1p2].data)), interpolation="nearest", cmap=plt.cm.gray)
                ax.set_title(p1p2)

        fig = plt.figure()
        i = 1
        for row in wp2.get_level(2, 'freq'):
            for node in row:
                ax = fig.add_subplot(len(row), len(row), i)
                ax.set_title("%s=(%s row, %s col)" % (
                             (node.path,) + wp2.expand_2d_path(node.path)))
                # ax.imshow(np.sqrt(np.abs(node.data)), origin='image', interpolation="nearest", cmap=plt.cm.gray)
                ax.imshow(np.sqrt(np.abs(node.data)), interpolation="nearest", cmap=plt.cm.gray)
                i += 1

        plt.show()
    # end def


    @staticmethod
    def wp_scalogram():
        import pywt
        x = np.linspace(0, 1, num=512)
        data = np.sin(250 * np.pi * x**2)

        wavelet = 'db2'
        level = 4
        order = "freq"  # other option is "normal"
        interpolation = 'nearest'
        cmap = plt.cm.cool

        # Construct wavelet packet
        wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
        nodes = wp.get_level(level, order=order)
        labels = [n.path for n in nodes]
        values = np.array([n.data for n in nodes], 'd')
        values = abs(values)

        # Show signal and wavelet packet coefficients
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("linchirp signal")
        ax.plot(x, data, 'b')
        ax.set_xlim(0, x[-1])

        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Wavelet packet coefficients at level %d" % level)
        # ax.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto", origin="image", extent=[0, 1, 0, len(values)])
        ax.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto", origin="lower", extent=[0, 1, 0, len(values)])
        ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels)

        # Show spectrogram and wavelet packet coefficients
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(211)
        ax2.specgram(data, NFFT=64, noverlap=32, Fs=2, cmap=cmap,
                     interpolation='bilinear')
        ax2.set_title("Spectrogram of signal")
        ax3 = fig2.add_subplot(212)
        ax3.imshow(values, origin='upper', extent=[-1, 1, -1, 1], interpolation='nearest')
        ax3.set_title("Wavelet packet coefficients")


        plt.show()
    # end def wp_scalogram

    @staticmethod
    def wp_visualize_coeffs_distribution():
        import pywt
        from pywt import WaveletPacket
        ecg = pywt.data.ecg()

        wp = WaveletPacket(ecg, 'sym5', maxlevel=4)

        fig = plt.figure()
        plt.set_cmap('bone')
        ax = fig.add_subplot(wp.maxlevel + 1, 1, 1)
        ax.plot(ecg, 'k')
        ax.set_xlim(0, len(ecg) - 1)
        ax.set_title("Wavelet packet coefficients")

        for level in range(1, wp.maxlevel + 1):
            ax = fig.add_subplot(wp.maxlevel + 1, 1, level + 1)
            nodes = wp.get_level(level, "freq")
            nodes.reverse()
            labels = [n.path for n in nodes]
            values = -abs(np.array([n.data for n in nodes]))
            ax.imshow(values, interpolation='nearest', aspect='auto')
            ax.set_yticks(np.arange(len(labels) - 0.5, -0.5, -1), labels)
            plt.setp(ax.get_xticklabels(), visible=False)

        plt.show()
    # end def
# end demo


def waveinfo(*argv):
    import pywt

    usage = """
    Usage:
        python waveinfo.py waveletname
        Example: python waveinfo.py 'sym5'
    """

    try:
        wavelet = pywt.Wavelet(argv[1])
        try:
            level = int(argv[2])
        except IndexError as e:    # analysis:ignore
            level = 10
    except ValueError as e:        # analysis:ignore
        print("Unknown wavelet")
        raise SystemExit
    except IndexError as e:        # analysis:ignore
        print(usage)
        raise SystemExit


    data = wavelet.wavefun(level)
    if len(data) == 2:
        x = data[1]
        psi = data[0]
        fig = plt.figure()
        if wavelet.complex_cwt:
            plt.subplot(211)
            plt.title(wavelet.name+' real part')
            mi, ma = np.real(psi).min(), np.real(psi).max()
            margin = (ma - mi) * 0.05
            plt.plot(x,np.real(psi))
            plt.ylim(mi - margin, ma + margin)
            plt.xlim(x[0], x[-1])
            plt.subplot(212)
            plt.title(wavelet.name+' imag part')
            mi, ma = np.imag(psi).min(), np.imag(psi).max()
            margin = (ma - mi) * 0.05
            plt.plot(x,np.imag(psi))
            plt.ylim(mi - margin, ma + margin)
            plt.xlim(x[0], x[-1])
        else:
            mi, ma = psi.min(), psi.max()
            margin = (ma - mi) * 0.05
            plt.plot(x,psi)
            plt.title(wavelet.name)
            plt.ylim(mi - margin, ma + margin)
            plt.xlim(x[0], x[-1])
    else:
        funcs, x = data[:-1], data[-1]
        labels = ["scaling function (phi)", "wavelet function (psi)",
                  "r. scaling function (phi)", "r. wavelet function (psi)"]
        colors = ("r", "g", "r", "g")
        fig = plt.figure()
        for i, (d, label, color) in enumerate(zip(funcs, labels, colors)):
            mi, ma = d.min(), d.max()
            margin = (ma - mi) * 0.05
            ax = fig.add_subplot((len(data) - 1) // 2, 2, 1 + i)

            ax.plot(x, d, color)
            ax.set_title(label)
            ax.set_ylim(mi - margin, ma + margin)
            ax.set_xlim(x[0], x[-1])

    plt.show()
# end def


# ========================================================================= #
# ========================================================================= #






if __name__=="__main__":
    # for testing the built-ins on your distribution:
    demo = demo()
    demo.cwt_analysis()
    demo.batch_processing()
    demo.benchmark()
    demo.dwt2_dwtn_image()
    demo.dwt_signal_decomposition()
    demo.dwt_swt_show_coeffs()
    demo.fswavedecn()
    demo.fswavedecn_mondrian()
    demo.plot_wavelets()
#    demo.plot_wavelets_pyqtgraph()   # no QT installed
    demo.swt2()
    demo.swt_variance()
    demo.wp_2d()
    demo.wp_scalogram()
    demo.wp_visualize_coeffs_distribution()

#    # waveinfo()


#    kpywt.sample()
# end if




# ========================================================================= #
# ========================================================================= #

