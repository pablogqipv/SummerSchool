# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:44:58 2017

@author: thpap
"""
import numpy as np


# =============================================================================
# Assign unknown NGA-West GMPE parameters using known ones according to Kaklamanos et al. (2011)
# =============================================================================

class unknownNGAparams():
    def __init__(self, Mag, SOF=None, azimuth=None, Z1=None, Fhw=None, rake=None, W=None, Z_tor=None, Zhyp=None,
                 delta=None, Rjb=None, R_rup=None, Rx=None, VS30=None):
        self.Mag = Mag
        self.SOF = SOF
        self.delta = delta
        self.Rjb = Rjb
        self.R_rup = R_rup
        self.Rx = Rx
        self.VS30 = VS30
        self.rake = rake
        self.Zhyp = Zhyp
        self.W = W
        self.Z_tor = Z_tor
        self.Fhw = Fhw
        self.azimuth = azimuth
        self.Z1 = Z1

    def SOF2rake(self):
        if self.SOF == 'Strike-Slip':
            self.rake = 0  # could also be 180...
        elif self.SOF == 'Reverse':
            self.rake = 90
        elif self.SOF == 'Normal':
            self.rake = -90
        else:
            raise SystemError('invalid SOF')

    def rake2SOF(self):
        ## ('Attention!!! rake2SOF per AS08 definitions')
        if 30 <= self.rake <= 150:
            self.SOF = 'Reverse'
        elif -120 <= self.rake <= -60:
            self.SOF = 'Normal'
        elif (-180 <= self.rake < -120) or (-60 < self.rake < 30) or (
                150 < self.rake <= 180):
            self.SOF = 'Strike-Slip'

    def SOF2delta(self):
        if self.SOF == 'Strike-Slip':
            self.delta = 90
        elif self.SOF == 'Reverse':
            self.delta = 40
        elif self.SOF == 'Normal':
            self.delta = 50
        else:
            raise SystemError('invalid SOF')

    def getW(self):
        if self.SOF == 'Strike-Slip':
            self.W = 10 ** (-0.76 + 0.27 * self.Mag)
        elif self.SOF == 'Reverse':
            self.W = 10 ** (-1.61 + 0.41 * self.Mag)
        elif self.SOF == 'Normal':
            self.W = 10 ** (-1.14 + 0.35 * self.Mag)
        else:
            raise SystemError('invalid SOF')

    def getZhyp(self):
        if self.SOF == 'Strike-Slip':
            self.Zhyp = 5.63 + 0.68 * self.Mag
        elif self.SOF == 'Reverse' or self.SOF == 'Normal':
            self.Zhyp = 11.24 - 0.2 * self.Mag
        else:
            self.Zhyp = 7.08 + 0.61 * self.Mag

    def getZ_tor(self):
        if self.delta is not None and self.Zhyp is not None and self.W is not None:
            self.Z_tor = max(self.Zhyp - 0.6 * self.W * np.sin(self.delta), 0)
        else:
            raise SystemError('something is missing')

    def getAzimuth(self):
        if self.Fhw == 1:
            self.azimuth = 50
        elif self.Fhw == 0:
            self.azimuth = -50
        else:
            raise SystemError('Fhw is missing')

    def Rjb2Rx(self):
        if self.delta == 90:
            self.Rx = self.Rjb * np.sin(self.azimuth)
        else:
            if self.Rjb == 0:  # there are better formulas upon knowledge of R_rup or Rx but I assume they are all unknown
                self.Rx = 0.5 * self.W * np.cos(self.delta)
            else:
                if (0 <= self.azimuth and self.azimuth < 90) or (90 < self.azimuth and self.azimuth <= 180):
                    if self.Rjb * abs(np.tan(self.azimuth)) <= self.W * np.cos(self.delta):
                        self.Rx = self.Rjb * abs(np.tan(self.azimuth))
                    elif self.Rjb * abs(np.tan(self.azimuth)) > self.W * np.cos(self.delta):
                        self.Rx = self.Rjb * np.tan(self.azimuth) * np.cos(self.azimuth - np.arcsin(
                            self.W * np.cos(self.delta) * np.cos(
                                self.azimuth) / self.Rjb))  # maybe not arcsin but sin^-1??
                elif self.azimuth == 90:
                    if self.Rjb > 0:
                        self.Rx = self.Rjb + self.W * np.cos(self.delta)
                elif -180 <= self.azimuth and self.azimuth < 0:
                    self.Rx = self.Rjb * np.sin(self.azimuth)
                else:
                    raise SystemError('Something went wrong!')

    def Rjb2R_rup(self):
        if self.delta == 90:
            self.R_rup = np.sqrt(self.Rjb ** 2 + self.Z_tor ** 2)
        else:
            if self.Rx < self.Z_tor * np.tan(self.delta):
                R_rup0 = np.sqrt(self.Rx ** 2 + self.Z_tor ** 2)
            elif self.Z_tor * np.tan(self.delta) <= self.Rx <= self.Z_tor * np.tan(self.delta) + self.W * (
                    1 / np.cos(self.delta)):
                R_rup0 = self.Rx * np.sin(self.delta) + self.Z_tor * np.cos(self.delta)
            elif self.Rx > self.Z_tor * np.tan(self.delta) + self.W * (1 / np.cos(self.delta)):
                R_rup0 = np.sqrt(
                    (self.Rx) - self.W * np.cos(self.delta) ** 2 + (self.Z_tor + self.W * np.sin(self.delta) ** 2))
            else:
                raise SystemError('Something went wrong!')

            if abs(self.azimuth) == 90:
                Ry = 0
            elif self.azimuth == 0 or abs(self.azimuth) == 180:
                Ry = self.Rjb
            else:
                Ry = abs(self.Rx * (1 / np.tan(self.azimuth)))

            self.R_rup = np.sqrt(R_rup0 ** 2 + Ry ** 2)

    def getZ1(self, GMPE):
        if GMPE == 'AS08':
            if self.VS30 < 180:
                self.Z1 = np.exp(6.745)
            elif self.VS30 >= 180 and self.VS30 <= 500:
                self.Z1 = np.exp(6.745 - 1.35 * np.log(self.VS30 / 180))
            else:
                self.Z1 = np.exp(5.394 - 4.48 * np.log(self.VS30 / 500))
        elif GMPE == 'CY08':
            self.Z1 = np.exp(28.5 - 3.82 * np.log((self.VS30 ** 8) + 378.7 ** 8) / 8)


def Rjb2Repi(m, rjb, SOF):
    """
    Scherbaum et al. (2004)
    Rjb to Repi
    mean values (std not implemented)
    i) strike-slip
    ii) normal or reverse
    iii) uknown (not implemented)
    """
    if SOF != 'Strike-Slip':
        if m < 6.75:
            a = np.array(
                [[-204.6322895121227, 134.154975829155, -31.88793034254544, 3.19291980403142, -0.1062775951463466],
                 [34.43763258518979, -24.6435799625101, 6.589592215019835, -0.7809972045047068, 0.03466381606160739],
                 [-0.3652835996406303, 0.2609425661042608, -0.06961489805608918, 0.008225795792183441,
                  -0.0003636289796775165],
                 [0.000964984017097691, -0.0006882375774797777, 0.0001832502549175116, -0.0000216016999879238,
                  9.52180898361203e-7]])
            muRepi = 0
            for row in range(1, 5):
                for col in range(1, 6):
                    muRepi += a[row - 1, col - 1] * (m ** (col - 1)) * (rjb ** (row - 1))
            return rjb + muRepi
        else:
            a = np.array(
                [[2393.700548190256, -1734.33155046077, 450.0764156086294, -50.43452786681108, 2.084201916889746],
                 [-289.1304106803179, 166.7731017148483, -35.91299876477036, 3.420547147077889, -0.1214755653713848],
                 [-3.079921992268037, 1.580845052618155, -0.3027669298725126, 0.02565255250122141,
                  -0.0008122570872540904],
                 [0.01514619493583962, -0.008008040071674015, 0.001584386532623823, -0.0001390707065706777,
                  4.572512707383002e-6]])
            muRepi = 0
            for row in range(1, 5):
                for col in range(1, 6):
                    muRepi += a[row - 1, col - 1] * (m ** (col - 1)) * (rjb ** (row - 1))
            return rjb + muRepi
    else:
        if m < 6.75:
            a = np.array(
                [[-505.9816209446427, 362.9789393165752, -96.7558292553837, 11.32583209792275, -0.4875149224839657],
                 [34.53144979329134, -25.06476703441657, 6.799388337927399, -0.8177184457906994, 0.03684000819116669],
                 [-0.3220626608065619, 0.2339810610103008, -0.06348397949783081, 0.007628838644194556,
                  -0.0003429950790833597],
                 [0.000809165697798322, -0.000588501060389523, 0.0001597758027600142, -0.00001920170447903924,
                  8.627613552893673e-7]])
            muRepi = 0
            for row in range(1, 5):
                for col in range(1, 6):
                    muRepi += a[row - 1, col - 1] * (m ** (col - 1)) * (rjb ** (row - 1))
            return rjb + muRepi
        else:
            a = np.array(
                [[-105280.2076341921, 56837.0555068667, -11473.76793935465, 1026.151410902254, -34.28770660852066],
                 [2141.279654008868, -1157.359557375134, 234.2191514134586, -21.03782861227021, 0.7078916320759653],
                 [-19.82771191674502, 10.78099383575376, -2.195276257330098, 0.1984260986892972, -0.006718910819424462],
                 [0.04726495858058512, -0.02581015595376117, 0.005278523087078716, -0.0004792030555530394,
                  0.0000162964723918607]])
            muRepi = 0
            for row in range(1, 5):
                for col in range(1, 6):
                    muRepi += a[row - 1, col - 1] * (m ** (col - 1)) * (rjb ** (row - 1))
            return rjb + muRepi

#
# aa = unknownNGAparams(7, SOF='Reverse', Fhw=0, Rjb=20)
# aa.SOF2rake()
# aa.SOF2delta()
# aa.getW()
# aa.getZhyp()
# aa.getZ_tor()
# aa.getAzimuth()
# aa.Rjb2Rx()
# aa.Rjb2R_rup()
