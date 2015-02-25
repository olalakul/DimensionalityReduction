# -*- coding: utf-8 -*-
"""
Given is the data frame (10x3578) with 10 spectra, 3578 (nonzero) measurements for each

Each spectrum (here also called signal) is denoised and encoded so that it is 
    described by a few hundred (depending on the denoising coefficient) nonzero values
Background can also be removed    
This technique can be used for any time series as dimentionaly reduction technique
Extracted nonzero values can be considered as nontrivial features and used further 
    in various machine learning algorithms
Similar techniques can be used (and are used) for image processing

@author: Olga Lalakulich olalakul AT gmail.com
python 3.4
File spectra_10.dat should be in the same directry as this file   
"""
#import os
#os.chdir("/home/olga/github/DimensionalityReduction")

import copy
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 180)
import pywt


def get_wavenumbers(spectra):
    """
    Extracts wavenumbers from column names of the pandas data frame
    Input:
    * spectra - pandas data frame with each spectrum along the row
    Output:
    * wavenumbers - numpy array of wavenumbers
    """
    lambda_max = float(spectra.columns[0][1:])  # maximal wavenumner in 0th column
    lambda_min = float(spectra.columns[-1][1:]) # minimal wavenumber in the last column
    wavenumbers = np.linspace(lambda_max, lambda_min, spectra.shape[1])
    return wavenumbers


def show_sample_spectra(spectra, samples=None, outF="sample_spectra.eps"):
    """
    Shows some spectra
    Input:
    * spectra - pandas data frame with each spectrum along the row
    * samples - list of samples to show, default is show all 
    * outF - file name to save the picture
    """
    if samples is None:
        samples = list(range(spectra.shape[0]))
    else:
        # check if all samples are in index 
        index_errors = set(samples).difference(set(spectra.index))
        if index_errors:
            samples = list(set(samples).intersection(set(spectra.index)))
            logger.warning("Samples %s are not within 0 and %s ... ignoring them",
                           ' '.join([str(i) for i in index_errors]), str(spectra.shape[0]) )
        
    logger.info("Making plot for samples %s", ' '.join([str(i) for i in index_errors]) )          
    wavenumbers = get_wavenumbers(spectra)
    plt.plot(wavenumbers, spectra.iloc[samples,:].transpose())
    plt.xlim([wavenumbers[0],wavenumbers[-1]])
    plt.xlabel('wavenumber [1/cm]')
    plt.ylabel('spectra of various samples')
    plt.text(7000, 2.3, str(spectra.shape[1])+" measurements for each sample", ha="left")
    plt.savefig(outF)
    logger.info("... saved to %s ... finished", outF )


def encode_signal(signal, denoising_coeff=5):
    """
    Reduces the number of nonzero values essential for describing of a signal
    Those can be used as nontrivial features for maching learning techniques
    Input:
    * signal - numpy array of real values
    * denoising_coeff - default 5, higher value means more smoothing
                        and fewer nonzero values in the encoded_signal
    Output:
    coded_signal (a list of numpy arrays) 
    """
    import pywt
    logger.info("Encoding signal ...")
    coded_signal = pywt.wavedec(signal, wavelet='sym8', mode='cpd')
    noiseSigma = denoising_coeff*np.std(coded_signal[-1]);
    threshold=noiseSigma*np.sqrt(2*np.log2(len(signal))); #print(threshold)
    # number of nonzero values is 10-20 times smaller than in the original signal
    coded_signal = list(map(lambda x: pywt.thresholding.soft(x,threshold),coded_signal))
    logger.info("... finished")
    return coded_signal


def remove_background(coded_signal):
    """
    Removes background from the coded signal
    Input:
    * coded_signal - as obtained from "encode_signal" function
    Output:
    * coded_signal_woBgr - ecoded signal (a list of numpy arrays) without background
    """
    import copy
    logger.info("Removing background ...")
    coded_signal_woBgr = copy.deepcopy(coded_signal)
    coded_signal_woBgr[0] = np.zeros(len(coded_signal[0]))
    logger.info("... finished")
    return coded_signal_woBgr

    
def reconstruct_signal(coded_signal):
    """
    Reconstructs signal from the coded signal
    Input:
    * coded_signal - as obtained from "encode_signal" function
    Output:
    * reconstructed - reconstructed signal (numpy array)
    """
    # reconstruct the signal
    import pywt
    logger.info("Reconstructing signal ...")
    reconstructed = pywt.waverec(coded_signal, wavelet='sym8', mode='cpd')
    logger.info("... finished")
    return reconstructed

    
def visualize_encoding_reconstruction(signal, denoising_coeff=5, wavenumbers=None,
                                      outF="reconstructing_signal.eps"):
    """
    Visualizes original signal, reconstructed signal and reconstructed signal without background
    Inputs:
    * signal - as in function "encode_signal"
    * denoising_coeff - as in function "encode_signal"
    * wavenumbers - wavenumber obtained from the "get_wavenumbers" function
    * outF - file name to save a picture with original and reconstructed signal
    """
    # encode signal
    coded_signal = encode_signal(signal, denoising_coeff=5)
    # count nonzero coefficients in the encoded signal
    num_nontrivial_features = sum([np.count_nonzero(w) for w in coded_signal])
    print("Encoding resulted in ", num_nontrivial_features, " nontrivial features")    
    # remove backgrond
    coded_signal_woBgr = remove_background(coded_signal)
    # reconstruct signal        
    reconstructed = reconstruct_signal(coded_signal)    
    # reconstruct signal  without background      
    reconstructed_woBgr = reconstruct_signal(coded_signal_woBgr)    
        
    # estimate deviations from the original signal
    print("Average deviation of the reconstructed signal from the original one",
          sum(abs(signal-reconstructed)/abs(signal))/len(signal)*100, "%")
    # maximum individual deviation
    print("Maximal deviation of the reconstructed signal from the original one",
          max(abs(signal-reconstructed)/abs(signal))*100, "%" )
          
    # plot original and reconstructed signals
    logger.info("Making plot for original and reconstructed signal")          
    if wavenumbers is None:
        wavenumbers = list(range(len(signal)))          
    df = pd.DataFrame({'original':signal, 
                       'reconstructed':reconstructed, 
                       'removed bgr':reconstructed_woBgr}, index=wavenumbers)
    df.plot(legend=True)
    plt.annotate("reconstructed: "+str(num_nontrivial_features)+" features out of "+str(len(signal)), xy=(0.4,0.95), xycoords='axes fraction')
    plt.annotate(str(num_nontrivial_features-len(coded_signal_woBgr[0]))+" after removing background", xy=(0.4,0.87), xycoords='axes fraction')
    plt.xlabel('wavenumber')
    plt.savefig(outF)
    logger.info("... saved to  %s  ... finished", outF)


    
if __name__ == "__main__":
    import os
    import sys
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    print(path)
    os.chdir(path)

    #logger
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # open file with 10 spectra, each as a row of pandas data frame
    spectra = pd.read_csv("spectra_10.dat")    
    
    # show available spectra
    show_sample_spectra(spectra, samples=[1,5,6,8,9], outF="sample_spectra.eps")
    #show_sample_spectra(spectra, outF="sample_spectra.eps")
    
    #take one signal from the available spectra to be used for encoding and reconstructing
    sample = 0
    signal = spectra.iloc[sample,:].values
    wavenumbers = get_wavenumbers(spectra)
    
    visualize_encoding_reconstruction(signal, denoising_coeff=5, wavenumbers=wavenumbers,
                                      outF="reconstructing_signal.eps")
    

    