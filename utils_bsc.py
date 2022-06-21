# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from pyproj import Proj, transform

from numba import njit
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import interpolate
import statsmodels.api as sm
import os
from openquake.hazardlib import gsim, imt, const
import numpy.matlib
from time import gmtime
from scipy.stats import skew
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
from UknownGMPEparams import unknownNGAparams

LonSite = 12.645
LatSite = 42.565

laea = {'proj': 'laea',
        'lat_0': LatSite,
        'lon_0': LonSite,
        'x_0': 0.,
        'y_0': 0.,
        'ellps': 'WGS84',
        'datum': 'WGS84'}
Proj_WGS84 = Proj({'init': 'epsg:4326'})
Proj_laea = Proj(laea)


def baker(T1, T2, orth=0):
    """
    References: Baker JW, Jayaram N. Correlation of Spectral Acceleration Values from NGA Ground Motion Models.
    Earthquake Spectra 2008; 24(1): 299–317. DOI: 10.1193/1.2857544.

    :param T1: First period
    :param T2: Second period
    :param orth: int, default is 0; 1 if the correlation coefficient is computed for the two orthogonal components
    :return: Predicted correlation coefficient
    """
    T_min = min(T1, T2)
    T_max = max(T1, T2)
    C1 = (1 - np.cos(np.pi / 2 - np.log(T_max / max(T_min, 0.109)) * 0.366));
    if T_max < 0.2:
        C2 = 1 - 0.105 * (1 - 1. / (1 + np.exp(100 * T_max - 5))) * (T_max - T_min) / (T_max - 0.0099);
    if T_max < 0.109:
        C3 = C2
    else:
        C3 = C1
    C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1 + np.cos(np.pi * (T_min) / (0.109)));

    if T_max <= 0.109:
        rho = C2
    elif T_min > 0.109:
        rho = C1
    elif T_max < 0.2:
        rho = min(C2, C4)
    else:
        rho = C4

    if orth:
        rho = rho * (0.79 - 0.023 * np.log(np.sqrt(t_min * t_max)))

    return rho


class CS:

    def __init__(self, mat_dir, T_star=0.5, gmpe='boore_atkinson_2008', pInfo=1):

        """
        T_star    : int, float, numpy.array, the default is None.
            Conditioning period or periods in case of AvgSa [sec].
        gmpe     : str, optional
            GMPE model (see OpenQuake library).
            The default is 'boore_atkinson_2008'.
        pInfo    : int, optional
            flag to print required input for the gmpe which is going to be used.
            (0: no, 1:yes)
            The default is 1.
        """

        # add T_star to self
        if isinstance(T_star, int) or isinstance(T_star, float):
            self.T_star = np.array([T_star])
        elif isinstance(T_star, numpy.ndarray):
            self.T_star = T_star
        # Add the input the ground motion database to use
        matfile = mat_dir
        self.database = loadmat(matfile, squeeze_me=True)
        # check if AvgSa or Sa is used as IM,
        # then in case of Sa(T*) add T* and Sa(T*) if not present
        if not self.T_star[0] in self.database['Periods'] and len(self.T_star) == 1:
            f = interpolate.interp1d(self.database['Periods'], self.database['Sa_rotD50'], axis=1)
            Sa_int = f(self.T_star[0])
            Sa_int.shape = (len(Sa_int), 1)
            Sa = np.append(self.database['Sa_rotD50'], Sa_int, axis=1)
            Periods = np.append(self.database['Periods'], self.T_star[0])
            self.database['Sa_rotD50'] = Sa[:, np.argsort(Periods)]
            self.database['Periods'] = Periods[np.argsort(Periods)]
        try:
            self.bgmpe = gsim.get_available_gsims()[gmpe]()
        except:
            raise KeyError('Not a valid gmpe')
        if pInfo == 1:  # print the selected gmpe info
            print('For the selected gmpe;')
            print(' The mandatory input distance parameters are %s' % list(self.bgmpe.REQUIRES_DISTANCES))
            print(' The mandatory input rupture parameters are %s' % list(self.bgmpe.REQUIRES_RUPTURE_PARAMETERS))
            print(' The mandatory input site parameters are %s' % list(self.bgmpe.REQUIRES_SITES_PARAMETERS))
            print(' The defined intensity measure component is %s' % self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT)
            print(' The defined tectonic region type is %s\n' % self.bgmpe.DEFINED_FOR_TECTONIC_REGION_TYPE)

    def Sa_avg(self, bgmpe, scenario, T):
        """
        GMPM of average spectral acceleration. The code will get as input the selected periods, Magnitude, distance and all
        other parameters of the selected GMPM (e.g. Boore & Atkinson, 2008) and will return the median and logarithmic
        spectral acceleration of the product of the Spectral accelerations at selected periods;


        """
        n = len(T)
        SPa_Med = np.zeros(n)
        SPa_STD = np.zeros(n)
        MoC = np.zeros((n, n))

        for i in range(n):
            SPa_Med[i], stddvs_lnSaT_star = bgmpe.get_mean_and_stddevs(scenario[0], scenario[1], scenario[2],
                                                                        imt.SA(period=T[i]), [const.StdDev.TOTAL])
            # convert to sigma_arb
            # One should uncomment this line if the arbitary component is used for
            # record selection.
            # ro_xy = 0.79-0.23*np.log(T[k])
            ro_xy = 1
            SPa_STD[i] = np.log(((np.exp(stddvs_lnSaT_star[0][0]) ** 2) * (2 / (1 + ro_xy))) ** 0.5)

            for j in range(n):
                rho = baker(T[i], T[j])
                MoC[i, j] = rho

        SPa_avg_meanLn = (1 / n) * sum(SPa_Med)  # logarithmic mean of Sa,avg
        SPa_avg_STD = 0
        for i in range(n):
            for j in range(n):
                SPa_avg_STD = SPa_avg_STD + (MoC[i][j] * SPa_STD[i] * SPa_STD[j])  # logarithmic Var of the Sa, avg
        SPa_avg_STD = (1 / n) ** 2 * SPa_avg_STD
        Sa = SPa_avg_meanLn
        sigma = SPa_avg_STD ** 0.5

        return Sa, sigma

    def rho_AvgSa_Sa(self, bgmpe, scenario, Ts, Tc_avg):

        """
        The correlation between Spectra acceleration and AvgSA

        :param Ts: these are all periods we are interested in
        :param T_r: these are the periods for which avgSa is calculated
        :param M:
        :param Rjb:
        :param Fault_Type:
        :param Vs30:
        :return: rho, predicted correlation coefficient
        """

        rho = 0
        for j in range(len(Tc_avg)):
            rho_bj = baker(Ts, Tc_avg[j])  # standard correlation for period T_r[j] with other periods
            _, UncSigmas = bgmpe.get_mean_and_stddevs(scenario[0], scenario[1], scenario[2],
                                                        imt.SA(period=Tc_avg[j]),
                                                        [const.StdDev.TOTAL])
            rho = (rho_bj * UncSigmas[0][0]) + rho

        _, Avg_sig = self.Sa_avg(bgmpe, scenario, Tc_avg)
        rho = rho / (len(Tc_avg) * Avg_sig)

        return rho

    def create(self, site_param={'vs30': 520}, rup_param={'rake': 0.0, 'mag': [7.2, 6.5]},
               dist_param={'rjb': [20, 5]}, Hcont=[0.6, 0.4], T_Tgt_range=[0.01, 4],
               im_T_star=1.0, epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram'):
        """
        Creates the target spectrum (conditional or unconditional).

        References
        ----------
        Baker JW. Conditional Mean Spectrum: Tool for Ground-Motion Selection.
        Journal of Structural Engineering 2011; 137(3): 322–331.
        DOI: 10.1061/(ASCE)ST.1943-541X.0000215.
        Kohrangi, M., Bazzurro, P., Vamvatsikos, D., and Spillatura, A.
        Conditional spectrum-based ground motion record selection using average
        spectral acceleration. Earthquake Engineering & Structural Dynamics,
        2017, 46(10): 1667–1685.

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


        Returns
        -------
        None.
        """

        # add target spectrum settings to self
        self.cond = cond
        self.useVar = useVar
        self.corr_func = corr_func

        if cond == 0:  # there is no conditioning period
            del self.T_star

        # Get number of scenarios, and their contribution
        nScenarios = len(rup_param['mag'])

        if Hcont is None:
            self.Hcont = [1 / nScenarios for _ in range(nScenarios)]
        else:
            self.Hcont = Hcont

        # Period range of the target spectrum
        temp = np.abs(self.database['Periods'] - np.min(T_Tgt_range))
        idx1 = np.where(temp == np.min(temp))[0][0]
        temp = np.abs(self.database['Periods'] - np.max(T_Tgt_range))
        idx2 = np.where(temp == np.min(temp))[0][0]
        T_Tgt = self.database['Periods'][idx1:idx2 + 1]

        # Get number of scenarios, and their contribution
        Hcont_mat = np.matlib.repmat(np.asarray(self.Hcont), len(T_Tgt), 1)

        # Conditional spectrum, log parameters
        TgtMean = np.zeros((len(T_Tgt), nScenarios))

        # co_variance
        TgtCov = np.zeros((nScenarios, len(T_Tgt), len(T_Tgt)))

        for n in range(nScenarios):
            # gmpe spectral values
            mu_lnSaT = np.zeros(len(T_Tgt))
            sigma_lnSaT = np.zeros(len(T_Tgt))

            # correlation coefficients
            rho_T_T_star = np.zeros(len(T_Tgt))

            # co_variance
            Cov = np.zeros((len(T_Tgt), len(T_Tgt)))

            # Set the contexts for the scenario
            site_param['sids'] = [0]  # This is required in OQ version 3.12.0
            sites = gsim.base.SitesContext()
            for key in site_param.keys():
                if key == 'rake':
                    site_param[key] = float(site_param[key])
                temp = np.array([site_param[key]])
                setattr(sites, key, temp)

            rup = gsim.base.RuptureContext()
            for key in rup_param.keys():
                if key == 'mag':
                    temp = np.array([rup_param[key][n]])
                else:
                    # temp = np.array([rup_param[key]])
                    temp = rup_param[key]
                setattr(rup, key, temp)

            dists = gsim.base.DistancesContext()
            for key in dist_param.keys():
                if key == 'rjb':
                    temp = np.array([dist_param[key][n]])
                else:
                    temp = np.array([dist_param[key]])
                setattr(dists, key, temp)

            scenario = [sites, rup, dists]
            # Calculate unconditional mean and std using the given GMPE
            for i in range(len(T_Tgt)):
                # Get the GMPE output for a rupture scenario
                mu0, sigma0 = self.bgmpe.get_mean_and_stddevs(sites, rup, dists, imt.SA(period=T_Tgt[i]),
                                                              [const.StdDev.TOTAL])
                mu_lnSaT[i] = mu0[0]
                sigma_lnSaT[i] = sigma0[0][0]

                if self.cond == 1:
                    # Compute the correlations between each T and T_star
                    rho_T_T_star[i] = self.rho_AvgSa_Sa(self.bgmpe, scenario, T_Tgt[i], self.T_star)

            if self.cond == 1:
                # Get the GMPE output and calculate Avg_Sa_T_star
                mu_lnSaT_star, sigma_lnSaT_star = self.Sa_avg(self.bgmpe, scenario, self.T_star)

                if epsilon is None:
                    # Back calculate epsilon
                    rup_eps = (np.log(im_T_star) - mu_lnSaT_star) / sigma_lnSaT_star
                else:
                    rup_eps = epsilon[n]

                # Get the value of the ln(CMS), conditioned on T_star
                TgtMean[:, n] = mu_lnSaT + rho_T_T_star * rup_eps * sigma_lnSaT

            elif self.cond == 0:
                TgtMean[:, n] = mu_lnSaT
            for i in range(len(T_Tgt)):
                for j in range(len(T_Tgt)):
                    var_1 = sigma_lnSaT[i] ** 2
                    var_2 = sigma_lnSaT[j] ** 2
                    # using Baker & Jayaram 2008 as correlation model
                    sigma_Corr = baker(T_Tgt[i], T_Tgt[j]) * np.sqrt(var_1 * var_2)
                    if self.cond == 1:
                        varT_star = sigma_lnSaT_star ** 2
                        sigma11 = np.matrix([[var_1, sigma_Corr], [sigma_Corr, var_2]])
                        sigma22 = np.array([varT_star])
                        sigma12 = np.array([rho_T_T_star[i] * np.sqrt(var_1 * varT_star),
                                            rho_T_T_star[j] * np.sqrt(varT_star * var_2)])
                        sigma12.shape = (2, 1)
                        sigma22.shape = (1, 1)
                        sigma_cond = sigma11 - sigma12 * 1. / sigma22 * sigma12.T
                        Cov[i, j] = sigma_cond[0, 1]
                    elif self.cond == 0:
                        Cov[i, j] = sigma_Corr
            # Get the value of standard deviation of target spectrum
            TgtCov[n, :, :] = Cov

        # over-write covariance matrix with zeros if no variance is desired in the ground motion selection
        if self.useVar == 0:
            TgtCov = np.zeros(TgtCov.shape)

        TgtMean_fin = np.sum(TgtMean * Hcont_mat, 1)
        # all 2D matrices are the same for each kk scenario, since sigma is only T dependent
        TgtCov_fin = TgtCov[0, :, :]
        Cov_elms = np.zeros((len(T_Tgt), nScenarios))
        for ii in range(len(T_Tgt)):
            for kk in range(nScenarios):
                # Hcont[kk] = contribution of the k-th scenario
                Cov_elms[ii, kk] = (TgtCov[kk, ii, ii] + (TgtMean[ii, kk] - TgtMean_fin[ii]) ** 2) * self.Hcont[kk]

        cov_diag = np.sum(Cov_elms, 1)
        TgtCov_fin[np.eye(len(T_Tgt)) == 1] = cov_diag

        # Find co_variance values of zero and set them to a small number so that
        # random number generation can be performed
        # TgtCov_fin[np.abs(TgtCov_fin) < 1e-10] = 1e-10
        min_eig = np.min(np.real(np.linalg.eigvals(TgtCov_fin)))
        if min_eig < 0:
            TgtCov_fin -= 10 * min_eig * np.eye(*TgtCov_fin.shape)

        TgtSigma_fin = np.sqrt(np.diagonal(TgtCov_fin))
        TgtSigma_fin[np.isnan(TgtSigma_fin)] = 0

        # Add target spectrum to self
        self.mu_ln = TgtMean_fin
        self.sigma_ln = TgtSigma_fin
        self.T = T_Tgt
        self.cov = TgtCov_fin

        if cond == 1:
            # add intensity measure level to self
            if epsilon is None:
                self.im_T_star = im_T_star
            else:
                f = interpolate.interp1d(self.T, np.exp(self.mu_ln))
                Sa_int = f(self.T_star)
                self.im_T_star = np.exp(np.sum(np.log(Sa_int)) / len(self.T_star))
                self.epsilon = epsilon

    def unconditional(self, site_param={'vs30': 520}, rup_param={'rake': 0.0, 'mag': [7.2, 6.5]},
                      dist_param={'rjb': [20, 5]}, Hcont=[0.6, 0.4], T_Tgt_range=[0.01, 4],
                      im_T_star=1.0, epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram'):
        """
        Creates the target spectrum (conditional or unconditional).

        References
        ----------
        Baker JW. Conditional Mean Spectrum: Tool for Ground-Motion Selection.
        Journal of Structural Engineering 2011; 137(3): 322–331.
        DOI: 10.1061/(ASCE)ST.1943-541X.0000215.
        Kohrangi, M., Bazzurro, P., Vamvatsikos, D., and Spillatura, A.
        Conditional spectrum-based ground motion record selection using average
        spectral acceleration. Earthquake Engineering & Structural Dynamics,
        2017, 46(10): 1667–1685.

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
        duration:

        Returns
        -------
        None.
        """

        # add target spectrum settings to self
        self.cond = cond
        self.useVar = useVar
        self.corr_func = corr_func

        if cond == 0:  # there is no conditioning period
            del self.T_star

        # Get number of scenarios, and their contribution
        nScenarios = len(rup_param['mag'])
        if Hcont is None:
            self.Hcont = [1 / nScenarios for _ in range(nScenarios)]
        else:
            self.Hcont = Hcont

        # Period range of the target spectrum
        temp = np.abs(self.database['Periods'] - np.min(T_Tgt_range))
        idx1 = np.where(temp == np.min(temp))[0][0]
        temp = np.abs(self.database['Periods'] - np.max(T_Tgt_range))
        idx2 = np.where(temp == np.min(temp))[0][0]
        T_Tgt = self.database['Periods'][idx1:idx2 + 1]

        # Get number of scenarios, and their contribution
        Hcont_mat = np.matlib.repmat(np.asarray(self.Hcont), len(T_Tgt), 1)

        # Conditional spectrum, log parameters
        TgtMean = np.zeros((len(T_Tgt), nScenarios))

        # co_variance
        TgtCov = np.zeros((nScenarios, len(T_Tgt), len(T_Tgt)))

        for n in range(nScenarios):
            # gmpe spectral values
            mu_lnSaT = np.zeros(len(T_Tgt))
            sigma_lnSaT = np.zeros(len(T_Tgt))

            # Set the contexts for the scenario
            site_param['sids'] = [0]  # This is required in OQ version 3.12.0
            sites = gsim.base.SitesContext()
            for key in site_param.keys():
                if key == 'rake':
                    site_param[key] = float(site_param[key])
                temp = np.array([site_param[key]])
                setattr(sites, key, temp)

            rup = gsim.base.RuptureContext()
            for key in rup_param.keys():
                if key == 'mag':
                    temp = np.array([rup_param[key][n]])
                else:
                    # temp = np.array([rup_param[key]])
                    temp = rup_param[key]
                setattr(rup, key, temp)

            dists = gsim.base.DistancesContext()
            for key in dist_param.keys():
                if key == 'rjb':
                    temp = np.array([dist_param[key][n]])
                else:
                    temp = np.array([dist_param[key]])
                setattr(dists, key, temp)

            scenario = [sites, rup, dists]
            # Calculate unconditional mean and std using the given GMPE
            for i in range(len(T_Tgt)):
                # Get the GMPE output for a rupture scenario
                mu0, sigma0 = self.bgmpe.get_mean_and_stddevs(sites, rup, dists, imt.SA(period=T_Tgt[i]),
                                                              [const.StdDev.TOTAL])
                mu_lnSaT[i] = mu0[0]
                sigma_lnSaT[i] = sigma0[0][0]

        return mu_lnSaT, sigma_lnSaT

    def plot_Target(self, j, save=0, show=0, name_dir='None'):
        """
        Plots the target spectrum

        """
        fig, ax = plt.subplots(figsize=(5.0, 4.6))
        ax.plot(self.T, np.exp(self.mu_ln), color='#ED5C8B', lw=1.2, alpha=0.95, label='mean')

        if self.useVar == 1:
            ax.fill_between(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), np.exp(self.mu_ln - 2 * self.sigma_ln),
                            alpha=0.35, color='lightgrey',
                            edgecolor="white")

        if self.cond == 1:
            if len(self.T_star) == 1:
                hatch = [float(self.T_star * 0.98), float(self.T_star * 1.02)]
                ax.axvspan(hatch[0], hatch[1], facecolor='darkgrey', alpha=0.5)
            else:
                hatch = [float(self.T_star.min()), float(self.T_star.max())]

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim([0.01, np.round(1.5 * np.max(np.exp(self.mu_ln + 2 * self.sigma_ln)), 1)])
        ax.set_xlim([0.1, self.T[-1]])
        ax.set_ylabel('Spectral acceleration [g]', fontsize=14)
        ax.set_xlabel('Period [s]', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.gca().spines["top"].set_alpha(.5)
        plt.gca().spines["bottom"].set_alpha(.5)
        plt.gca().spines["right"].set_alpha(.5)
        plt.gca().spines["left"].set_alpha(.5)
        plt.tight_layout()

        if save == 1:
            plt.savefig(name_dir + "//" + "Target_mean_IM_" + str(j + 1) + ".png", dpi=800)
        if show == 1:
            plt.show()

        # Sample and target standard deviations
        if self.useVar == 1:
            fig, ax = plt.subplots(figsize=(5.0, 4.6))
            ax.plot(self.T, self.sigma_ln, color='#ED5C8B', lw=1.2, label='Target dispersion')
            if self.cond == 1:
                if len(self.T_star) == 1:
                    hatch = [float(self.T_star * 0.98), float(self.T_star * 1.02)]
                else:
                    hatch = [float(self.T_star.min()), float(self.T_star.max())]
                ax.axvspan(hatch[0], hatch[1], facecolor='darkgrey', alpha=0.5)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_ylim([0, 1.5 * np.max(self.sigma_ln, axis=0)])
            ax.set_xlim([0.1, self.T[-1]])
            ax.set_ylabel('Dispersion', fontsize=14)
            ax.set_xlabel('Period [s]', fontsize=14)
            ax.set_xscale('log')
            plt.gca().spines["top"].set_alpha(.5)
            plt.gca().spines["bottom"].set_alpha(.5)
            plt.gca().spines["right"].set_alpha(.5)
            plt.gca().spines["left"].set_alpha(.5)
            plt.tick_params(axis="x", labelsize=12)
            plt.tick_params(axis="y", labelsize=12)
            plt.tight_layout()
            if save == 1:
                plt.savefig(name_dir + "//" + "Target_std_IM_" + str(j + 1) + ".png", dpi=800)
            if show == 1:
                plt.show()

    def getUnknown_params(self, M, Fhw, rake, Rjb, Vs30):

        kaklam = unknownNGAparams(Mag=M, SOF=None, azimuth=None, Z1=None, Fhw=Fhw, rake=rake, W=None, Z_tor=None,
                                  Zhyp=None, delta=None, Rjb=Rjb, R_rup=None, Rx=None, VS30=Vs30)
        kaklam.rake2SOF()
        kaklam.SOF2delta()
        kaklam.getW()
        kaklam.getZhyp()
        kaklam.getZ_tor()
        kaklam.getAzimuth()
        kaklam.Rjb2Rx()
        kaklam.Rjb2R_rup()
        kaklam.getZ1(GMPE='AS08')
        SOF = kaklam.SOF
        W = kaklam.W
        dip = kaklam.delta
        Ztor = kaklam.Z_tor
        Rrup = kaklam.R_rup
        Rx = kaklam.Rx
        Ry0 = -999  # unknown
        Z1 = kaklam.Z1

        return SOF, W, dip, Ztor, Rrup, Rx, Ry0, Z1

    def search_database(self):
        """
        Details
        -------
        Search the database and does the filtering.

        Parameters
        ----------
        None.

        Returns
        -------
        sampleBig : numpy.array
            An array which contains the IMLs from filtered database.
        soil_Vs30 : numpy.array
            An array which contains the Vs30s from filtered database.
        magnitude : numpy.array
            An array which contains the magnitudes from filtered database.
        Rjb : numpy.array
            An array which contains the Rjbs from filtered database.
        mechanism : numpy.array
            An array which contains the fault type info from filtered database.
        Filename_1 : numpy.array
            An array which contains the filename of 1st gm component from filtered database.
            If selection is set to 1, it will include filenames of both components.
        Filename_2 : numpy.array
            An array which contains the filenameof 2nd gm component filtered database.
            If selection is set to 1, it will be None value.
        NGA_num : numpy.array
            If NGA_W2 is used as record database, record sequence numbers from filtered
            database will be saved, for other databases this variable is None.
        """

        if self.selection == 1:  # SaKnown = Sa_arb
            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            soil_Vs30 = np.append(self.database['Vs30'], self.database['Vs30'], axis=0)
            Mw = np.append(self.database['Magnitude'], self.database['Magnitude'], axis=0)
            Rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
            Mw_type = np.append(self.database['Magnitude_type'], self.database['Magnitude_type'], axis=0)
            SD_5_75_H1 = np.append(self.database['D5_75_H1'], self.database['D5_75_H2'], axis=0)
            Filename_1 = np.append(self.database['FileName_1'], self.database['FileName_2'], axis=0)
            fault = np.array([0] * len(Filename_1))
            NGA_num = np.array([0] * len(Filename_1))
            eq_ID = np.array([0] * len(Filename_1))
            DB = np.append(self.database['DB'], self.database['DB'], axis=0)
            dt = np.append(self.database['DT'], self.database['DT'], axis=0)
            nstp = np.append(self.database['NSTP'], self.database['NSTP'], axis=0)

        elif self.selection == 2:  # SaKnown = Sa_g.m. or RotD50
            if self.Sa_def == 'GeoMean':
                SaKnown = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.Sa_def == 'RotD50':  # SaKnown = Sa_RotD50.
                SaKnown = self.database['Sa_rotD50']
            else:
                raise ValueError('Unexpected Sa definition, exiting...')
            soil_Vs30 = self.database['Vs30']
            Mw = self.database['Magnitude']
            Rjb = self.database['Rjb']
            Mw_type = self.database['Magnitude_type']
            SD_5_75_H1 = self.database['D5_75_H1']
            SD_5_75_H2 = self.database['D5_75_H2']
            Filename_1 = self.database['FileName_1']
            Filename_2 = self.database['FileName_2']
            Proximity = self.database['proximity_code']
            fault = np.array([0] * len(Filename_2))
            NGA_num = np.array([0] * len(Filename_2))
            eq_ID = np.array([0] * len(Filename_2))
            DB = self.database['DB']
            dt = self.database['DT']
            nstp = self.database['NSTP']
        else:
            raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')

        perKnown = self.database['Periods']

        # Limiting the records to be considered using the `notAllowed' variable
        # Sa cannot be negative or zero, remove these.
        notAllowed = np.unique(np.where(SaKnown <= 0)[0]).tolist()

        if self.Vs30_lim is not None:  # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim) * np.invert(np.isnan(soil_Vs30)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Mw_lim is not None:  # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim) * np.invert(np.isnan(Mw)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Rjb_lim is not None:  # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim) * np.invert(np.isnan(Rjb)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.fault_lim is not None:  # limiting values on mechanism exist
            mask = (fault == self.fault_lim * np.invert(np.isnan(fault)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.DB_lim is not None:  # limiting values on DB in mixed case
            if (len(self.DB_lim)) == 1:
                mask = (DB != self.DB_lim)
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
            elif (len(self.DB_lim)) == 2:
                mask = (DB != self.DB_lim[0])
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
                mask = (DB != self.DB_lim[1])
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
            elif (len(self.DB_lim)) == 3:
                mask = (DB != self.DB_lim[0])
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
                mask = (DB != self.DB_lim[1])
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
                mask = (DB != self.DB_lim[2])
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
            else:
                print("Smth wrong with DB limit!!!")

        # keep only free field motions (or close to structure but not inside the structure)
        if self.freefield is True:
            # keeping all that are not in array N_free
            N_free = ['C', 'D', 'E', 'F', 'G', 'P', 'Q', 'R', 'S', 'T', '3.0']
            for an in N_free:
                mask = Proximity != an
                temp = [i for i, x in enumerate(mask) if not x]
                notAllowed.extend(temp)
        # get the unique values
        notAllowed = (list(set(notAllowed)))
        Allowed = [i for i in range(SaKnown.shape[0])]
        for i in notAllowed:
            Allowed.remove(i)
        # Use only allowed records
        SaKnown = SaKnown[Allowed, :]
        soil_Vs30 = soil_Vs30[Allowed]
        Mw = Mw[Allowed]
        Rjb = Rjb[Allowed]
        fault = fault[Allowed]
        DB = DB[Allowed]
        dt = dt[Allowed]
        nstp = nstp[Allowed]
        Mw_type = Mw_type[Allowed]
        SD_5_75_H1 = SD_5_75_H1[Allowed]
        Filename_1 = Filename_1[Allowed]
        Proximity = Proximity[Allowed]

        if self.selection == 1:
            Filename_2 = None
            SD_5_75_H2 = None
        else:
            Filename_2 = Filename_2[Allowed]
            SD_5_75_H2 = SD_5_75_H2[Allowed]
        # Arrange the available spectra in a usable format and check for invalid input
        # Match periods (known periods and periods for error computations)
        recPer = []
        for i in range(len(self.T)):
            recPer.append(np.where(perKnown == self.T[i])[0][0])

        # Check for invalid input
        sampleBig = SaKnown[:, recPer]
        if np.any(np.isnan(sampleBig)):
            raise ValueError('NaNs found in input response spectra')

        if self.nGM > len(Filename_1):
            raise ValueError('There are not enough records which satisfy',
                             'the given record selection criteria...',
                             'Please use broaden your selection criteria...')
        return sampleBig, soil_Vs30, Mw, Mw_type, Rjb, fault, Filename_1, Filename_2, eq_ID, DB, dt, nstp, SD_5_75_H1, SD_5_75_H2, Proximity

    def simulate_spectra(self):
        """
        Generates simulated response spectra with best matches to the target values.
        """

        # Set initial seed for simulation
        if self.seedValue != 0:
            np.random.seed(0)
        else:
            np.random.seed(sum(gmtime()[:6]))

        devTotalSim = np.zeros((self.nTrials, 1))
        specDict = {}
        nT = len(self.T)
        # Generate simulated response spectra with best matches to the target values
        for j in range(self.nTrials):
            specDict[j] = np.zeros((self.nGM, nT))
            for i in range(self.nGM):
                # Note: we may use latin hypercube sampling here instead. I leave it as Monte Carlo for now
                specDict[j][i, :] = np.exp(np.random.multivariate_normal(self.mu_ln, self.cov))

            devMeanSim = np.mean(np.log(specDict[j]),
                                 axis=0) - self.mu_ln  # how close is the mean of the spectra to the target
            devSigSim = np.std(np.log(specDict[j]),
                               axis=0) - self.sigma_ln  # how close is the mean of the spectra to the target
            devSkewSim = skew(np.log(specDict[j]),
                              axis=0)  # how close is the skewness of the spectra to zero (i.e., the target)

            devTotalSim[j] = self.weights[0] * np.sum(devMeanSim ** 2) + \
                             self.weights[1] * np.sum(devSigSim ** 2) + \
                             0.1 * (self.weights[2]) * np.sum(
                devSkewSim ** 2)  # combine the three error metrics to compute a total error

        recUse = np.argmin(np.abs(devTotalSim))  # find the simulated spectra that best match the targets
        self.sim_spec = np.log(specDict[recUse])  # return the best set of simulations

    def select(self, nGM=30, selection=1, Sa_def='RotD50', isScaled=1, maxScale=4, minScale=0, Mw_lim=None, DB_lim=None,
               Vs30_lim=None, Rjb_lim=None, fault_lim=None, freefield=False, nTrials=20, weights=[1, 2, 0.3],
               seedValue=0,
               nLoop=2, penalty=0, tol=10):
        """
         Perform the ground motion selection.

         References
         ----------
         Jayaram, N., Lin, T., and Baker, J. W. (2011).
         A computationally efficient ground-motion selection algorithm for
         matching a target response spectrum mean and variance.
         Earthquake Spectra, 27(3), 797-815.


         Parameters
         ----------
         nGM : int, optional, the default is 30.
             Number of ground motions to be selected.
         selection : int, optional, The default is 1.
             1 for single-component selection and arbitrary component sigma.
             2 for two-component selection and average component sigma.
         Sa_def : str, optional, the default is 'RotD50'.
             The spectra definition. Necessary if selection = 2.
             'GeoMean' or 'RotD50'.
         isScaled : int, optional, the default is 1.
             0 not to allow use of amplitude scaling for spectral matching.
             1 to allow use of amplitude scaling for spectral matching.
         maxScale : float, optional, the default is 4.
             The maximum allowable scale factor
         Mw_lim : list, optional, the default is None.
             The limiting values on magnitude.
         Vs30_lim : list, optional, the default is None.
             The limiting values on Vs30.
         Rjb_lim : list, optional, the default is None.
             The limiting values on Rjb.
         fault_lim : int, optional, the default is None.
             The limiting fault mechanism.
             0 for unspecified fault
             1 for strike-slip fault
             2 for normal fault
             3 for reverse fault
         seedValue  : int, optional, the default is 0.
             For repeatability. For a particular seedValue not equal to
             zero, the code will output the same set of ground motions.
             The set will change when the seedValue changes. If set to
             zero, the code randomizes the algorithm and different sets of
             ground motions (satisfying the target mean and variance) are
             generated each time.
         weights : numpy.array or list, optional, the default is [1,2,0.3].
             Weights for error in mean, standard deviation and skewness
         nTrials : int, optional, the default is 20.
             nTrials sets of response spectra are simulated and the best set (in terms of
             matching means, variances and skewness is chosen as the seed). The user
             can also optionally rerun this segment multiple times before deciding to
             proceed with the rest of the algorithm. It is to be noted, however, that
             the greedy improvement technique significantly improves the match between
             the means and the variances subsequently.
         nLoop   : int, optional, the default is 2.
             Number of loops of optimization to perform.
         penalty : int, optional, the default is 0.
             > 0 to penalize selected spectra more than
             3 sigma from the target at any period, = 0 otherwise.
         tol     : int, optional, the default is 10.
             Tolerable percent error to skip optimization

         Returns
         -------
         None.
         """

        # Add selection settings to self
        self.nGM = nGM
        self.selection = selection
        self.Sa_def = Sa_def
        self.isScaled = isScaled
        self.Mw_lim = Mw_lim
        self.DB_lim = DB_lim
        self.Vs30_lim = Vs30_lim
        self.Rjb_lim = Rjb_lim
        self.fault_lim = fault_lim
        self.seedValue = seedValue
        self.weights = weights
        self.nTrials = nTrials
        self.maxScale = maxScale
        self.minScale = minScale
        self.nLoop = nLoop
        self.tol = tol
        self.penalty = penalty
        self.freefield = freefield

        # Simulate response spectra
        self.simulate_spectra()
        # Search the database and filter
        sampleBig, Vs30, Mw, Mw_type, Rjb, fault, Filename_1, Filename_2, eq_ID, DB, dt, nstp, SD_5_75_H1, SD_5_75_H2, Proximity = self.search_database()
        # Processing available spectra
        sampleBig = np.log(sampleBig)
        nBig = sampleBig.shape[0]

        # Find best matches to the simulated spectra from ground-motion database
        recID = np.ones(self.nGM, dtype=int) * (-1)
        finalScaleFac = np.ones(self.nGM)
        sampleSmall = np.ones((self.nGM, sampleBig.shape[1]))
        weights = np.array(weights)

        if self.cond == 1 and self.isScaled == 1:
            # Calculate IMLs for the sample
            f = interpolate.interp1d(self.T, np.exp(sampleBig), axis=1)
            sampleBig_imls = np.exp(np.sum(np.log(f(self.T_star)), axis=1) / len(self.T_star))

        if self.cond == 1 and len(self.T_star) == 1:
            # These indices are required in case IM = Sa(T) to break the loop
            ind2 = (np.where(self.T != self.T_star[0])[0]).tolist()

        # Find nGM ground motions, initial subset
        for i in range(self.nGM):
            err = np.zeros(nBig)
            scaleFac = np.ones(nBig)
            # Calculate the scaling factor
            if self.isScaled == 1:
                # using conditioning IML
                if self.cond == 1:
                    scaleFac = self.im_T_star / sampleBig_imls
                # using error minimization
                elif self.cond == 0:
                    scaleFac = np.sum(np.exp(sampleBig) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(
                        np.exp(sampleBig) ** 2, axis=1)
            else:
                scaleFac = np.ones(nBig)

            mask1 = scaleFac > self.maxScale
            mask2 = scaleFac < self.minScale
            mask = np.logical_xor(mask1, mask2)
            idxs = np.where(~mask)[0]
            err[mask] = 1000000
            err[~mask] = np.sum((np.log(
                np.exp(sampleBig[idxs, :]) * scaleFac[~mask].reshape(len(scaleFac[~mask]), 1)) -
                                 self.sim_spec[i, :]) ** 2, axis=1)
            # to avoid repeating the same records
            for iii in recID:
                if iii > 0:
                    err[iii] = 1000000

            recID[i] = int(np.argsort(err)[0])  # record with the minimal error

            if err.min() >= 1000000:
                raise Warning('     Possible problem with simulated spectrum. No good matches found')

            if self.isScaled == 1:
                finalScaleFac[i] = scaleFac[recID[i]]

            # Save the selected spectra
            sampleSmall[i, :] = np.log(np.exp(sampleBig[recID[i], :]) * finalScaleFac[i])

        self.rec_initial_set = sampleSmall  # save the initial set to the self
        # error of the initial set

        if self.cond == 1 and len(self.T_star) == 1:  # if conditioned on SaT, ignore error at T*
            medianErr = np.max(
                np.abs(np.exp(np.mean(sampleSmall[:, ind2], axis=0)) - np.exp(self.mu_ln[ind2])) / np.exp(
                    self.mu_ln[ind2])) * 100
            stdErr = np.max(
                np.abs(np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) / self.sigma_ln[ind2]) * 100
            if self.useVar == 1:
                SSE_s = weights[0] * np.sum(((np.mean(sampleSmall[:, ind2], axis=0) - self.mu_ln[ind2]) ** 2)) + \
                    weights[1] * np.sum(((np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) ** 2))
            else:
                SSE_s = np.sum(((np.mean(sampleSmall[:, ind2], axis=0) - self.mu_ln[ind2]) ** 2))

        else:
            medianErr = np.max(
                np.abs(np.exp(np.mean(sampleSmall, axis=0)) - np.exp(self.mu_ln)) / np.exp(
                    self.mu_ln)) * 100
            stdErr = np.max(
                np.abs(np.std(sampleSmall, axis=0) - self.sigma_ln) / self.sigma_ln) * 100
            if self.useVar == 1:
                SSE_s = weights[0] * np.sum(((np.mean(sampleSmall, axis=0) - self.mu_ln) ** 2)) + \
                    weights[1] * np.sum(((np.std(sampleSmall, axis=0) - self.sigma_ln) ** 2))
            else:
                SSE_s = np.sum(((np.mean(sampleSmall, axis=0) - self.mu_ln) ** 2))

        SSE_print = 100 * SSE_s
        print("Before greedy optimization")
        print(' Max error in median = %.2f %%' % medianErr)
        print(' Max error in standard deviation = %.2f %%' % stdErr)
        print(' SSE_s = %.2f %%' % SSE_print)


        recID = recID.tolist()

        if self.useVar == 1:
            # Apply Greedy subset modification procedure
            # Use njit to speed up the optimization algorithm
            print("Starting Greedy Optimization")

            @njit
            def find_rec(sampleSmall, scaleFac, mu_ln, sigma_ln, recIDs):

                def mean_numba(a):

                    res = []
                    for i in range(a.shape[1]):
                        res.append(a[:, i].mean())

                    return np.array(res)

                def std_numba(a):

                    res = []
                    for i in range(a.shape[1]):
                        res.append(a[:, i].std())

                    return np.array(res)

                minDev = 100000
                for j in range(nBig):
                    if not np.any(recIDs == j):
                        # Add to the sample the scaled spectra
                        temp = np.zeros((1, len(sampleBig[j, :])))
                        temp[:, :] = sampleBig[j, :]
                        tempSample = np.concatenate((sampleSmall, temp + np.log(scaleFac[j])), axis=0)
                        devMean = mean_numba(tempSample) - mu_ln  # Compute deviations from target
                        devSig = std_numba(tempSample) - sigma_ln
                        devTotal = weights[0] * np.sum(devMean * devMean) + weights[1] * np.sum(
                            devSig * devSig)  # this is SSEs

                        # Check if we exceed the scaling limit and if the record is already in the set
                        if scaleFac[j] > maxScale or np.any(recIDs == j) or scaleFac[j] < minScale:
                            devTotal = devTotal + 1000000
                        # Penalize bad spectra
                        elif penalty > 0:
                            for m in range(nGM):
                                devTotal = devTotal + np.sum(
                                    np.abs(np.exp(tempSample[m, :]) > np.exp(mu_ln + 3.0 * sigma_ln))) * penalty
                                devTotal = devTotal + np.sum(
                                    np.abs(np.exp(tempSample[m, :]) < np.exp(mu_ln - 3.0 * sigma_ln))) * penalty
                        # Should cause improvement and record should not be repeated
                        if devTotal < minDev:
                            minID = j
                            minDev = devTotal
                return minID

            for k in range(self.nLoop):  # Number of passes
                for i in range(self.nGM):  # Loop for nGM
                    sampleSmall = np.delete(sampleSmall, i, 0)  # it deletes record i
                    oldrec = recID[i]
                    recID = np.delete(recID, i)
                    # Calculate the scaling factor
                    if self.isScaled == 1:
                        # using conditioning IML
                        if self.cond == 1:
                            scaleFac = self.im_T_star / sampleBig_imls
                        # using error minimization
                        elif self.cond == 0:
                            scaleFac = np.sum(np.exp(sampleBig) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(
                                np.exp(sampleBig) ** 2, axis=1)
                    else:
                        scaleFac = np.ones(nBig)

                    # Try to add a new spectra to the subset list

                    minID = find_rec(sampleSmall, scaleFac, self.mu_ln, self.sigma_ln, recID)

                    sampleSmall_new = np.concatenate(
                        (sampleSmall[:i, :],
                         sampleBig[minID, :].reshape(1, sampleBig.shape[1]) + np.log(scaleFac[minID]),
                         sampleSmall[i:, :]), axis=0)
                    recID_new = np.concatenate((recID[:i], np.array([minID]), recID[i:]))
                    if self.cond == 1 and len(self.T_star) == 1:
                        SSE_s_new = weights[0] * np.sum(
                            ((np.mean(sampleSmall_new[:, ind2], axis=0) - self.mu_ln[ind2]) ** 2)) + \
                                    weights[1] * np.sum(
                            ((np.std(sampleSmall_new[:, ind2], axis=0) - self.sigma_ln[ind2]) ** 2))
                    else:
                        SSE_s_new = weights[0] * np.sum(
                            ((np.mean(sampleSmall_new, axis=0) - self.mu_ln) ** 2)) + \
                                    weights[1] * np.sum(
                            ((np.std(sampleSmall_new, axis=0) - self.sigma_ln) ** 2))

                    if SSE_s_new < SSE_s:  # if it is improved
                        if self.isScaled == 1:
                            finalScaleFac[i] = scaleFac[minID]
                        else:
                            finalScaleFac[i] = 1
                        sampleSmall = sampleSmall_new
                        recID = recID_new
                        SSE_s = SSE_s_new
                    else:
                        minID = oldrec
                        sampleSmall = np.concatenate((sampleSmall[:i, :],
                                                      sampleBig[minID, :].reshape(1, sampleBig.shape[1]) + np.log(
                                                          finalScaleFac[i]), sampleSmall[i:, :]), axis=0)
                        recID = np.concatenate((recID[:i], np.array([minID]), recID[i:]))
                        if self.cond == 1 and len(self.T_star) == 1:
                            SSE_s = weights[0] * np.sum(((np.mean(sampleSmall[:, ind2], axis=0) - self.mu_ln[ind2]) ** 2)) + \
                                    weights[1] * np.sum(((np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) ** 2))
                        else:
                            SSE_s = weights[0] * np.sum(((np.mean(sampleSmall, axis=0) - self.mu_ln) ** 2)) + \
                                    weights[1] * np.sum(((np.std(sampleSmall, axis=0) - self.sigma_ln) ** 2))

                # Lets check if the selected ground motions are good enough, if the errors are sufficiently small stop!

                if self.cond == 1 and len(self.T_star) == 1:
                    medianErr = np.max(
                        np.abs(np.exp(np.mean(sampleSmall[:, ind2], axis=0)) - np.exp(self.mu_ln[ind2])) / np.exp(
                            self.mu_ln[ind2])) * 100
                    stdErr = np.max(
                        np.abs(np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) / self.sigma_ln[ind2]) * 100
                    SSE_s = weights[0] * np.sum(100 * ((np.mean(sampleSmall[:, ind2], axis=0) - self.mu_ln[ind2]) ** 2)) + \
                        weights[1] * np.sum(100 * ((np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) ** 2))
                else:
                    medianErr = np.max(
                        np.abs(np.exp(np.mean(sampleSmall, axis=0)) - np.exp(self.mu_ln)) / np.exp(
                            self.mu_ln)) * 100
                    stdErr = np.max(
                        np.abs(np.std(sampleSmall, axis=0) - self.sigma_ln) / self.sigma_ln) * 100
                    SSE_s = weights[0] * np.sum(100 * ((np.mean(sampleSmall, axis=0) - self.mu_ln) ** 2)) + \
                        weights[1] * np.sum(100 * ((np.std(sampleSmall, axis=0) - self.sigma_ln) ** 2))

                if medianErr < self.tol and stdErr < self.tol and SSE_s < 10:
                    break
            print('After greedy optimization:')
            print(' Max error in median = %.2f %%' % medianErr)
            print(' Max error in standard deviation = %.2f %%' % stdErr)
            print(' SSE_s = %.2f %%' % SSE_s)
        if self.useVar == 1:
            # Calculate the d-statistics of the final set
            KS = []
            for i in range(len(self.T)):
                if self.cond == 1 and len(self.T_star) == 1:
                    if self.T[i] == self.T_star[0]:
                        KS.append(0)
                else:
                    f1 = sampleSmall[:, i]
                    f1 = np.sort(f1)
                    f1 = np.concatenate([[min(f1)], f1])
                    emp_cdf = np.linspace(0, 1, (self.nGM) + 1)
                    norm_cdf = st.norm.cdf(f1, self.mu_ln[i], self.sigma_ln[i])
                    KS.append((max(np.abs(emp_cdf - norm_cdf))))
            self.KS = KS




        # Add selected record information to self

        self.rec_scale = finalScaleFac
        self.rec_spec = sampleSmall
        self.rec_Vs30 = Vs30[recID]
        self.rec_Rjb = Rjb[recID]
        self.rec_Mw = Mw[recID]
        self.rec_fault = fault[recID]
        self.eq_ID = eq_ID[recID]
        self.rec_h1 = Filename_1[recID]
        self.rec_h2 = Filename_2[recID]
        self.proximity = Proximity[recID]
        self.SSE_s = SSE_s
        self.DB = DB[recID]
        self.dt = dt[recID]
        self.nstp = nstp[recID]
        self.sampleBig = sampleBig
        self.rec_Mw_type = Mw_type[recID]
        self.rec_SD_5_75_H1 = SD_5_75_H1[recID]

        if self.selection == 1:
            self.rec_h2 = None
            self.rec_SD_5_75_H2 = None
        elif self.selection == 2:
            self.rec_h2 = Filename_2[recID]
            self.rec_SD_5_75_H2 = SD_5_75_H2[recID]




    def plot_sel(self, j, save=0, show=0, initial=0, name_dir='None'):
        """
        Plots the selected

        """
        # Plot Target spectrum vs. mean selected response spectra
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot(self.T, np.exp(self.mu_ln), color='#ED5C8B', lw=1.2, alpha=0.95, label='Target')
        ax.plot(self.T, np.exp(np.mean(self.rec_spec, axis=0)), color='#47DBCD', lw=1.2, label='Selected set')
        if self.useVar == 1:
            ax.plot(self.T, np.exp(np.mean(self.rec_initial_set, axis=0)), color='silver', lw=1.2, label='Initial set')

        if self.useVar == 1:
            ax.plot(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='#ED5C8B', linestyle="dashed")
            ax.plot(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='#ED5C8B', linestyle="dashed")
            ax.plot(self.T, np.exp(np.mean(self.rec_spec, axis=0) + 2 * np.std(self.rec_spec, axis=0)),
                    color='#47DBCD', linestyle='--', lw=1.2)
            ax.plot(self.T, np.exp(np.mean(self.rec_spec, axis=0) - 2 * np.std(self.rec_spec, axis=0)),
                    color='#47DBCD', linestyle='--', lw=1.2)
            ax.plot(self.T, np.exp(np.mean(self.rec_initial_set, axis=0) + 2 * np.std(self.rec_initial_set, axis=0)),
                    color='silver', linestyle='--', lw=1.2)
            ax.plot(self.T, np.exp(np.mean(self.rec_initial_set, axis=0) - 2 * np.std(self.rec_initial_set, axis=0)),
                    color='silver', linestyle='--', lw=1.2)

        plt.legend(loc='upper right', fontsize=12.0, frameon=False)
        if self.cond == 1:
            if len(self.T_star) == 1:
                hatch = [float(self.T_star * 0.98), float(self.T_star * 1.02)]
            else:
                hatch = [float(self.T_star.min()), float(self.T_star.max())]
            ax.axvspan(hatch[0], hatch[1], facecolor='darkgrey', alpha=0.5)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # ax.set_xticks(np.arange(1, (1 + nIM)))
        ax.set_ylim([0.01, np.round(1.5 * np.max(np.exp(self.mu_ln + 2 * self.sigma_ln)), 1)])
        ax.set_xlim([0.1, self.T[-1]])
        ax.set_ylabel('Spectral acceleration [g]', fontsize=14)
        ax.set_xlabel('Period [s]', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.gca().spines["top"].set_alpha(.5)
        plt.gca().spines["bottom"].set_alpha(.5)
        plt.gca().spines["right"].set_alpha(.5)
        plt.gca().spines["left"].set_alpha(.5)

        #  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #  ncol=2, mode="expand", borderaxespad=0, fontsize=8, frameon=False, prop={'size': 7})
        plt.tight_layout()
        if save == 1:
            plt.savefig(name_dir + "//" + "Selected_IM_" + str(j + 1) + ".png", dpi=800)
        if show == 1:
            plt.show()

        # Sample and target standard deviations
        if self.useVar == 1:
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            ax.plot(self.T, self.sigma_ln, color='#ED5C8B', lw=1.2, label='Target')
            ax.plot(self.T, np.std(self.rec_spec, axis=0), color='#47DBCD', lw=1.2, label='Selected set')
            ax.plot(self.T, np.std(self.rec_initial_set, axis=0), color='silver', lw=1.2, label='Initial set')

            if self.cond == 1:
                if len(self.T_star) == 1:
                    hatch = [float(self.T_star * 0.98), float(self.T_star * 1.02)]
                else:
                    hatch = [float(self.T_star.min()), float(self.T_star.max())]
                ax.axvspan(hatch[0], hatch[1], facecolor='darkgrey', alpha=0.5)

            ax.tick_params(axis='y', labelsize=7)
            # ax.set_xticks(np.arange(1, (1 + nIM)))
            ax.set_ylim([0, 1.5 * np.max(self.sigma_ln, axis=0)])
            ax.set_xlim([0.1, self.T[-1]])
            ax.set_ylabel('Dispersion', fontsize=14)
            ax.set_xlabel('Period [s]', fontsize=14)
            ax.set_xscale('log')
            plt.legend(loc='upper right', fontsize=12.0, frameon=False)
            plt.gca().spines["top"].set_alpha(.5)
            plt.gca().spines["bottom"].set_alpha(.5)
            plt.gca().spines["right"].set_alpha(.5)
            plt.gca().spines["left"].set_alpha(.5)
            plt.tick_params(axis="x", labelsize=8)
            plt.tick_params(axis="y", labelsize=12)
            plt.tight_layout()
            if save == 1:
                plt.savefig(name_dir + "//" + "Selected_std_IM_" + str(j + 1) + ".png", dpi=800)
            if show == 1:
                plt.show()
