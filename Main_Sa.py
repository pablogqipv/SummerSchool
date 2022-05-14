"""
Script used to select the records based on the Sa(T*) value. It is based (with some modifications) on the toolbox made
    by Volkan: https://github.com/volkanozsarac/EzGM.git;

Calculation of SSEs is from "A Computationally Efficient Ground-Motion Selection Algorithm for Matching a Target Response
    Spectrum Mean and Variance" Jayaram;

Any GMPE available in the OQ library can be used. If we set pinfo=1 we will get information about GMPE (e.g. for which
    IM it was derived)
make sure that you use same GMPE in hazard calculation (in OQ) and for the selection
mat file needs to be in [g]
        Parameters
        ----------
        site_param : dictionary
            Contains required site parameters to define target spectrum.
        rup_param  : dictionary
            Contains required rupture parameters to define target spectrum.
        dist_param : dictionary
            Contains required distance parameters to define target spectrum.
        Hcont      : list, optional, the default is None.
            Hazard contribution for considered scenarios.
            If None hazard contribution is the same for all scenarios.
        im_T_star   : int, float, optional, the default is 1.
            Conditioning intensity measure level [g] (conditional selection)
        epsilon    : list, optional, the default is None.
            Epsilon values for considered scenarios (conditional selection)
        T_Tgt_range: list, optional, the default is [0.01,4].
            Lower and upper bound values for the period range of target spectrum.
        cond       : int, optional
            0 to run unconditional selection
            1 to run conditional selection
        useVar     : int, optional, the default is 1.
            0 not to use variance in target spectrum
            1 to use variance in target spectrum
        corr_func: str, optional, the default is baker_jayaram
            correlation model to use "baker_jayaram","akkar"
"""
import numpy as np
from utils_bsc import CS
from time import time
from OQProc import disagg_MReps, hazard
import os
import pickle
from matplotlib import pyplot as plt
import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import warnings
warnings.filterwarnings("ignore")


def saveMat(OutFolderName, name_mat):
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

    for im in range(nIM):
        Fname1.append(All_CS[im].rec_h1)
        Fname2.append(All_CS[im].rec_h2)
        meanReq.append(All_CS[im].mu_ln)
        SF.append(All_CS[im].rec_scale)
        sampleSmall.append(All_CS[im].rec_spec)
        covReq_fin.append(All_CS[im].cov)
        SSE_all.append(All_CS[im].SSE_s)
        DB_all.append(All_CS[im].DB)
        nstp_sel.append(All_CS[im].nstp)
        dt_sel.append(All_CS[im].dt)
        Mw_sel.append(All_CS[im].rec_Mw)
        Rjb_sel.append(All_CS[im].rec_Rjb)
        Vs30_sel.append(All_CS[im].rec_Vs30)
        Mw_type.append(All_CS[im].rec_Mw_type)
        SD_5_75_H1.append(All_CS[im].rec_SD_5_75_H1)
        SD_5_75_H2.append(All_CS[im].rec_SD_5_75_H2)

    scipy.io.savemat(name_mat, {'Fname1': Fname1, 'Fname2': Fname2, 'PerTgt': All_CS[0].T, 'meanReq': meanReq, 'SF': SF,
                                'sampleSmall': sampleSmall, 'covReq_fin': covReq_fin, 'SSE': SSE_all, 'DB_sel': DB_all,
                                'nstp_sel': nstp_sel, 'dt_sel': dt_sel, 'Mw_type': Mw_type, 'Mw_sel': Mw_sel,
                                'Rjb_sel': Rjb_sel, 'Vs30_sel': Vs30_sel, 'SD_5_75_H1': SD_5_75_H1,
                                'SD_5_75_H2': SD_5_75_H2})


# INPUT

poes = [0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0.015, 0.01, 0.006, 0.002]  # probability of exceedance in 50 years
T_target = [0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.50, 1.6,
            1.7, 1.8, 1.9, 2.00, 3.00, 4.00]  # periods of interest
fig_dir = r"D:\Documents\PhD\METIS\Summer_school\Summer_school\Figures"  # directory where the figures are stored
output_dir = r"D:\Documents\PhD\METIS\Summer_school\Summer_school\output"  # directory where pkl and mat are saved
mat_dir = r"D:\Documents\PhD\METIS\Rock v soil\Perugia\record_selection\DB_final_6.mat"  # mat file with records
path_dis = r"D:\Documents\PhD\METIS\Rock v soil\Record Selection\Hazard_SaT\Perugia_disagg\Py_1"  # output from OQ
path_haz = r"D:\Documents\PhD\METIS\Rock v soil\Record Selection\Hazard_SaT\output_N\Py_1"  # output from OQ
nIM = 1  # number of IM levels
T_star = 1  # period of interest
Mbin = 0.5  # the values used in hazard calculation (OQ)
dbin = 10  # the values used in hazard calculation (OQ)

# Input for the GMPE (depends on the GMPE)
rake = 180
Vs30 = 800
Fhw = 1

# RUNNING ANALYSIS
print("Reading values from the disaggregation")
# Read the results of the disaggregation and get M, R and epsilon values
meanLst, modeLst, _, _, _ = disagg_MReps(Mbin, dbin, poes, path_dis, fig_dir, n_rows=3,
                                iplot=True)  # mean and mode scenarios from disaggregation (M, R and epsilon)
imls = hazard(poes, path_haz, output_dir=fig_dir,
              rlz='hazard_curve-mean', i_save=1, i_show=1)  # IMs at the predefined levels (poes)
All_CS = []
for j in range(nIM):
    print("IM level:", str(1 + j))
    # Initialize the CS class
    CS1 = CS(mat_dir, T_star=T_star, gmpe='BooreAtkinson2008', pInfo=1)
    mag = meanLst[j][0]
    rjb = meanLst[j][1]
    eps = meanLst[j][2]  # not used here
    im_T_star = imls[0][j]
    print("Poe is:", poes[j], "in 50 years.", "Mean Magnitude and Rjb are:", np.round(mag, 2), np.round(rjb, 2))
    print("IM at the level", str(j+1), "is:", np.round(im_T_star, 3))
    SOF, W, dip, Ztor, Rrup, Rx, Ryo, Z1 = CS1.getUnknown_params(M=mag, Fhw=Fhw, rake=rake, Rjb=rjb, Vs30=Vs30)
    # Create the target spectra
    CS1.create(site_param={'vs30': Vs30},
                                    rup_param={'rake': rake, 'mag': [mag]},
                                    dist_param={'rjb': [rjb], 'rrup': Rrup, 'rx': Rx, 'ry0': Ryo}, Hcont=None, T_Tgt_range=T_target, im_T_star=im_T_star,
                                    epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram')
    print("Target distribution is defined")

    CS1.plot_Target(j, save=0, show=1, name_dir=fig_dir)
    # Select ground motion records
    CS1.select(nGM=40, selection=2, Sa_def='RotD50', isScaled=1, maxScale=5, minScale=0, Mw_lim=[4, 8],
               DB_lim=['Ridge'], Vs30_lim=[300, 2000], Rjb_lim=None, fault_lim=None, freefield=True, nTrials=20,
               weights=[1, 2, 0.3], seedValue=0, nLoop=20, penalty=1, tol=10)

    CS1.plot_sel(j, save=1, show=1, initial=1, name_dir=fig_dir)  # initial is 1 if we want to plot before greedy
    All_CS.append(CS1)

# Save the results
os.chdir(output_dir)
name = "Records.pkl"
with open(name, 'wb') as file:
    pickle.dump(All_CS, file)
# A new file will be created
name_mat = "Records.mat"
saveMat(output_dir, name_mat)
