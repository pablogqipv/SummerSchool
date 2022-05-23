"""
OpenQuake PSHA Post-Processing ToolBox

"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm  # import colormap
from matplotlib.patches import Patch
import numpy as np
import numpy.matlib
import pandas as pd
from scipy import interpolate


def hazard(poes, path_hazard_results, output_dir='Outputs', rlz='hazard_curve-mean', i_save=0, i_show=0):
    """
    Details
    -------
    This script will plot the hazard curve

    Parameters
    ----------
    poes : list
        Probabilities of exceedance in tw years for which im levels will be obtained.
    path_hazard_results: str
        Path to the hazard results
    output_dir: str, optional
        Save outputs
    rlz : str, optional
        realization name to plot.

    Returns
    -------
    None.

    """

    # Initialise some lists
    lat = []
    lon = []
    im = []
    s = []
    poe = []
    apoe = []
    id_no = []
    imls = []

    # Read through each file in the outputs folder
    count = 0
    for file in os.listdir(path_hazard_results):
        if file.startswith(rlz):
            # Strip the IM out of the file name
            im_type = (file.rsplit('-')[2]).rsplit('_')[0]
            # Load the results in as a dataframe
            df = pd.read_csv(''.join([path_hazard_results, '/', file]), skiprows=2)
            # Get the column headers (but they have a 'poe-' string in them to strip out)
            iml = list(df.columns.values)[3:]  # List of headers
            iml = [float(i[4:]) for i in iml]  # Strip out the actual IM values
            f = open(''.join([path_hazard_results, '/', file]), "r")
            temp1 = f.readline().split(',')
            temp2 = list(filter(None, temp1))
            inv_t = float(temp2[5].replace(" investigation_time=", ""))
            f.close()

            # Append each site's info to the output array
            lat.append([df.lat[0]][0])
            lon.append([df.lon[0]][0])
            im.append(im_type)
            s.append(iml)
            # Get the array of poe in inv_t
            poe.append(df.iloc[0, 3:].values)

            # For each array of poe, convert it to annual poe
            temp = []
            for i in np.arange(len(poe[-1])):
                temp.append(-np.log(1 - poe[-1][i]) / inv_t)
            apoe.append(temp)
    # Get intensity measure levels corresponding to poes
    for i in range(len(s)):
        plt.figure(figsize=(6.4, 5.2))
        plt.loglog(s[i], poe[i], label=im[i], color='salmon', lw=1, alpha=0.95)
        Ninterp = 1e5
        iml_range = np.arange(min(s[i]), max(s[i]), (max(s[i]) - min(s[i])) / Ninterp)
        poe_fit = interpolate.interp1d(s[i], np.asarray(poe[i]), kind='quadratic')(iml_range)
        idxs = []
        for ij in range(len(poes)):
            temp = abs(poe_fit - poes[ij]).tolist()
            idxs.append(temp.index(min(temp)))
            # These are actual points where the analysis are carried out and losses are calculated for
        iml = iml_range[idxs]
        imls.append(iml)
        plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname': 'Times New Roman'}
        plt.tick_params(axis="x", labelsize=11)
        plt.tick_params(axis="y", labelsize=11)
        plt.ylim([0.001, 2])
        plt.xlim([0.01, 3])
        plt.xlabel(str(im[i]) + '  [g]', fontsize=14, **csfont)
        plt.ylabel('Probability of exceedance in 50 years', fontsize=14, **csfont)
        plt.gca().spines["top"].set_alpha(.3)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.3)
        plt.gca().spines["left"].set_alpha(.3)
        plt.grid(True, which="both", color='gainsboro', alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(output_dir, 'Hazard_Curve_' + im[i] + '.png')
        if i_save == 1:
            plt.savefig(fname, format='png', dpi=900)
        if i_show == 1:
            plt.show()
    return imls


def disagg_MReps(Mbin, dbin, poe_disagg, path_disagg_results, output_dir, n_rows=1, iplot=False):
    """
    This scripts reads the results of the disaggregation

    Parameters
    ----------
    poe_disagg : list
        disaggregation probability of exceedances
    path_disagg_results: str
        Path to the hazard results
        :param iplot:
        :param Mbin:
        :param dbin:
        :param n_rows:


    """
    cmap = cm.get_cmap('gnuplot')  # Get desired colormap
    lat = []
    lon = []
    modeLst, meanLst = [], []
    im = []
    poe = []
    Tr = []
    apoe_norm = []
    M, R, eps = [], [], []
    probs = []
    mags = []
    dists = []

    for file in os.listdir(path_disagg_results):
        if file.startswith('rlz') and file.find('Mag_Dist_Eps') > 0:
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)
            # Strip the IM out of the file name
            im.append(file.rsplit('-')[2])
            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            try:  # for OQ version <3.11
                inv_t = float(ff[9].replace(" investigation_time=", ""))
                poe.append(float(ff[12].replace(" poe=", "").replace("'", "")))
            except:
                inv_t = float(ff[6].replace(" investigation_time=", ""))
                poe.append(float(ff[-1].replace(" poe=", "").replace("\"", "").replace("\n", "")))
            lon.append(float(ff[10].replace(" lon=", "")))
            lat.append(float(ff[11].replace(" lat=", "")))
            Tr.append(-inv_t / np.log(1 - poe[-1]))

            # Extract the poe and annualise
            df['apoe'] = -np.log(1 - df['poe']) / inv_t

            # Normalise the apoe for disaggregation plotting
            df['apoe_norm'] = df['apoe'] / df['apoe'].sum()
            apoe_norm.append(df['apoe_norm'])

            # Compute the modal value (highest apoe)
            mode = df.sort_values(by='apoe_norm', ascending=False)[0:1]
            modeLst.append([mode['mag'].values[0], mode['dist'].values[0], mode['eps'].values[0]])

            # Compute the mean value
            meanLst.append([np.sum(df['mag'] * df['apoe_norm']), np.sum(df['dist'] * df['apoe_norm']),
                            np.sum(df['eps'] * df['apoe_norm'])])

            M.append(df['mag'])
            R.append(df['dist'])
            eps.append(df['eps'])
            probs.append(df['poe'])

    lon = [x for _, x in sorted(zip(Tr, lon))]
    lat = [x for _, x in sorted(zip(Tr, lat))]
    im = [x for _, x in sorted(zip(Tr, im))]
    M = [x for _, x in sorted(zip(Tr, M))]
    R = [x for _, x in sorted(zip(Tr, R))]
    eps = [x for _, x in sorted(zip(Tr, eps))]
    apoe_norm = [x for _, x in sorted(zip(Tr, apoe_norm))]
    modeLst = [x for _, x in sorted(zip(Tr, modeLst))]
    meanLst = [x for _, x in sorted(zip(Tr, meanLst))]

    Tr = -inv_t / np.log(1 - np.asarray(poe_disagg))
    n_Tr = len(np.unique(np.asarray(Tr)))
    Tr = sorted(Tr)
    ims = np.unique(im)
    n_im = len(ims)
    n_eps = len(np.unique(np.asarray(eps)))
    min_eps = np.min(np.unique(np.asarray(eps)))  # get range of colorbars so we can normalize
    max_eps = np.max(np.unique(np.asarray(eps)))

    lon = lon[0]
    lat = lat[0]

    n_cols = int(np.floor(n_Tr / n_rows))
    if np.mod(n_Tr, n_rows):
        n_cols += 1
    if iplot:
        if n_im>1:
            for idx1 in range(n_im):
                fig = plt.figure(figsize=(19.2, 10.8))
                for idx2 in range(n_Tr):
                    i = idx1 * n_Tr + idx2
                    ax1 = fig.add_subplot(n_rows, n_cols, idx2 + 1, projection='3d')

                    # scale each eps to [0,1], and get their rgb values
                    rgba = [cmap((k - min_eps) / max_eps / 2) for k in (np.unique(np.asarray(eps)))]
                    num_triads_M_R_eps = len(R[i])
                    Z = np.zeros(int(num_triads_M_R_eps / n_eps))

                    for l in range(n_eps):
                        X = np.array(R[i][np.arange(l, num_triads_M_R_eps, n_eps)])
                        Y = np.array(M[i][np.arange(l, num_triads_M_R_eps, n_eps)])

                        dx = np.ones(int(num_triads_M_R_eps / n_eps)) * dbin / 2
                        dy = np.ones(int(num_triads_M_R_eps / n_eps)) * Mbin / 2
                        dz = np.array(apoe_norm[i][np.arange(l, num_triads_M_R_eps, n_eps)]) * 100

                        ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba[l], zsort='average', alpha=0.7, shade=True)
                        Z += dz  # add the height of each bar to know where to start the next

                    ax1.set_xlabel('Rjb [km]')
                    ax1.set_ylabel('$M_{w}$')
                    if np.mod(idx2 + 1, n_cols) == 1:
                        ax1.set_zlabel('Hazard Contribution [%]')
                        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                        ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
                    ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

                    plt.title(
                        '$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km, $\epsilon_{mod}$=%s\n$M_{mean}$=%s, $R_{mean}$=%s '
                        'km, $\epsilon_{mean}$=%s'
                        % ("{:.0f}".format(Tr[i]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                        "{:.1f}".format(modeLst[i][2]),
                        "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1]), "{:.1f}".format(meanLst[i][2])),
                        fontsize=11, loc='right', va='top', y=0.95)

                    mags.append(meanLst[i][0])
                    dists.append(meanLst[i][1])

                legend_elements = []
                for j in range(n_eps):
                    legend_elements.append(Patch(facecolor=rgba[n_eps - j - 1],
                                                label='\u03B5 = %.2f' % (np.unique(np.asarray(eps))[n_eps - j - 1])))

                fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.05), borderaxespad=0.,
                        ncol=n_eps)
                plt.subplots_adjust(hspace=0.05, wspace=0.05)  # adjust the subplot to the right for the legend
                fig.suptitle('Disaggregation of Seismic Hazard\nIntensity Measure: %s\nLatitude: %s, Longitude: %s' % (
                    ims[idx1], "{:.2f}".format(lat), "{:.2f}".format(lon)), fontsize=14, weight='bold', ha='left', x=0.12,
                            y=0.97)
                fname = os.path.join(output_dir, 'Disaggregation_MReps_' + ims[idx1] + '.png')
                plt.savefig(fname, format='png', dpi=600)
                plt.show()

    return meanLst, modeLst, M, R, probs
