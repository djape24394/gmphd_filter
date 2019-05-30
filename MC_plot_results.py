import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import time
import ospa
import pickle

if __name__=='__main__':
    with open('MC2ospatnum500.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        (osp_all_total, osp_loc_total, osp_card_total, tnum_total) = pickle.load(f)

    ospall = np.array(osp_all_total)
    ospAllMean = ospall.mean(0)
    osploc = np.array(osp_loc_total)
    ospLocMean = osploc.mean(0)
    ospcar = np.array(osp_card_total)
    ospCarMean = ospcar.mean(0)
    tnum = np.array(tnum_total)
    tnumMean = tnum.mean(0)
    tnumStd = tnum.std(0)

    plt.figure()
    plt.plot(ospAllMean)

    plt.figure()
    plt.plot(ospLocMean)

    plt.figure()
    plt.plot(ospCarMean)

    plt.figure()
    plt.plot(tnumMean)
    plt.plot(tnumMean - tnumStd)
    plt.plot(tnumMean + tnumStd)