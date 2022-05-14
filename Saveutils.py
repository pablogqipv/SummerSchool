# -*- coding: utf-8 -*-
"""

"""

import os
import scipy
import warnings

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