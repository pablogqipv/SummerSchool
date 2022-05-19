# -*- coding: utf-8 -*-
"""

"""

import os
import scipy
import warnings
from scipy.io import loadmat
import numpy as np

from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import interpolate



def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import warnings
warnings.filterwarnings("ignore")


def saveMat(OutFolderName, name_mat,nIM,All_CS):
    """
    Saving output as mat file that can be used with Matlab SDOFs
    you need to go back to this
    :return:
    """
    os.chdir(OutFolderName)
    import scipy
    Fname1 = []
    Fname2 = []
    meanReq = []
    SF = []
    sampleSmall = []
    covReq_fin = []
    SSE_all = []
    DB_all = []
    nstp_sel = []
    dt_sel = []
    Mw_sel = []
    Rjb_sel = []
    Vs30_sel = []
    Mw_type = []
    SD_5_75_H2 = []
    SD_5_75_H1 = []

    # for im in range(nIM):
    #     Fname1.append(All_CS[im].rec_h1)
    #     Fname2.append(All_CS[im].rec_h2)
    #     meanReq.append(All_CS[im].mu_ln)
    #     SF.append(All_CS[im].rec_scale)
    #     sampleSmall.append(All_CS[im].rec_spec)
    #     covReq_fin.append(All_CS[im].cov)
    #     SSE_all.append(All_CS[im].SSE_s)
    #     DB_all.append(All_CS[im].DB)
    #     nstp_sel.append(All_CS[im].nstp)
    #     dt_sel.append(All_CS[im].dt)
    #     Mw_sel.append(All_CS[im].rec_Mw)
    #     Rjb_sel.append(All_CS[im].rec_Rjb)
    #     Vs30_sel.append(All_CS[im].rec_Vs30)
    #     Mw_type.append(All_CS[im].rec_Mw_type)
    #     SD_5_75_H1.append(All_CS[im].rec_SD_5_75_H1)
    #     SD_5_75_H2.append(All_CS[im].rec_SD_5_75_H2)
        
    # for im in range(nIM):
    Fname1.append(All_CS.rec_h1)
    Fname2.append(All_CS.rec_h2)
    meanReq.append(All_CS.mu_ln)
    SF.append(All_CS.rec_scale)
    sampleSmall.append(All_CS.rec_spec)
    covReq_fin.append(All_CS.cov)
    SSE_all.append(All_CS.SSE_s)
    DB_all.append(All_CS.DB)
    nstp_sel.append(All_CS.nstp)
    dt_sel.append(All_CS.dt)
    Mw_sel.append(All_CS.rec_Mw)
    Rjb_sel.append(All_CS.rec_Rjb)
    Vs30_sel.append(All_CS.rec_Vs30)
    Mw_type.append(All_CS.rec_Mw_type)
    SD_5_75_H1.append(All_CS.rec_SD_5_75_H1)
    SD_5_75_H2.append(All_CS.rec_SD_5_75_H2)    

    scipy.io.savemat(name_mat, {'Fname1': Fname1, 'Fname2': Fname2, 'PerTgt': All_CS[0].T, 'meanReq': meanReq, 'SF': SF,
                                'sampleSmall': sampleSmall, 'covReq_fin': covReq_fin, 'SSE': SSE_all, 'DB_sel': DB_all,
                                'nstp_sel': nstp_sel, 'dt_sel': dt_sel, 'Mw_type': Mw_type, 'Mw_sel': Mw_sel,
                                'Rjb_sel': Rjb_sel, 'Vs30_sel': Vs30_sel, 'SD_5_75_H1': SD_5_75_H1,
                                'SD_5_75_H2': SD_5_75_H2})
    
def plot_select_recs(All_CS,mat_dir,T_star):
    Fname1 = []
    Fname2 = []
    meanReq = []
    covReq_fin = []
    SF = []
    Fname1.append(All_CS[0].rec_h1)
    Fname2.append(All_CS[0].rec_h2)
    meanReq.append(All_CS[0].mu_ln) 
    SF.append(All_CS[0].rec_scale)
    covReq_fin.append(All_CS[0].cov)
    matfile = mat_dir
    database = loadmat(matfile, squeeze_me=True)
    if not T_star[0] in database['Periods'] and len(T_star) == 1:
        f = interpolate.interp1d(database['Periods'], database['Sa_rotD50'], axis=1)
        Sa_int = f(T_star[0])
        Sa_int.shape = (len(Sa_int), 1)
        Sa = np.append(database['Sa_rotD50'], Sa_int, axis=1)
        Periods = np.append(database['Periods'], T_star[0])
        database['Sa_rotD50'] = Sa[:, np.argsort(Periods)]
        database['Periods'] = Periods[np.argsort(Periods)]
        
    for k in Fname1:
        seq_num=database['FileName_1']==k
        print(seq_num)