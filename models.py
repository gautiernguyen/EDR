import numpy as np

def MP_Lin2010(phi_in,th_in,Pd,Pm,Bz,tilt=0.):
    ''' The Lin 2010 Magnetopause model. Returns the MP distance for a given
    azimuth (phi), zenith (th), solar wind dynamic and magnetic pressures (nPa)
    and Bz (in nT).
    * th: Zenith angle from positive x axis (zenith) (between 0 and pi)
    * phi: Azimuth angle from y axis, about x axis (between 0 abd 2 pi)
    * Pd: Solar wind dynamic pressure in nPa
    * Pm: Solar wind magnetic pressure in nPa
    * tilt: Dipole tilt
    '''
    a = [12.544,
         -0.194,
         0.305,
         0.0573,
         2.178,
         0.0571,
         -0.999,
         16.473,
         0.00152,
         0.382,
         0.0431,
         -0.00763,
         -0.210,
         0.0405,
         -4.430,
         -0.636,
         -2.600,
         0.832,
         -5.328,
         1.103,
         -0.907,
         1.450]

    arr = type(np.array([]))

    if(type(th_in) == arr):
        th = th_in.copy()
    else:
        th = th_in

    if(type(phi_in) == arr):
        phi = phi_in.copy()
    else:
        phi = phi_in

    el = th_in < 0.
    if(type(el) == arr):
        if(el.any()):
            th[el] = -th[el]

            if(type(phi) == type(arr)):
                phi[el] = phi[el]+np.pi
            else:
                phi = phi*np.ones(th.shape)+np.pi*el
    else:
        if(el):
            th = -th
            phi = phi+np.pi

    P = Pd+Pm

    def exp2(i):
        return a[i]*(np.exp(a[i+1]*Bz)-1)/(np.exp(a[i+2]*Bz)+1)

    def quad(i, s):
        return a[i]+s[0]*a[i+1]*tilt+s[1]*a[i+2]*tilt**2

    r0 = a[0]*P**a[1]*(1+exp2(2))

    beta = [a[6] + exp2(7),
            a[10],
            quad(11, [1, 0]),
            a[13]]

    f = np.cos(0.5*th)+a[5]*np.sin(2*th)*(1-np.exp(-th))
    s = beta[0]+beta[1]*np.cos(phi)+beta[2]*np.sin(phi)+beta[3]*np.sin(phi)**2
    f = f**(s)

    c = {}
    d = {}
    TH = {}
    PHI = {}
    e = {}
    for i, s in zip(['n', 's'], [1, -1]):
        c[i] = a[14]*P**a[15]
        d[i] = quad(16, [s, 1])
        TH[i] = quad(19, [s, 0])
        PHI[i] = np.cos(th)*np.cos(TH[i])
        PHI[i] = PHI[i] + np.sin(th)*np.sin(TH[i])*np.cos(phi-(2-s)*0.5*np.pi)
        PHI[i] = np.arccos(PHI[i])
        e[i] = a[21]
    r = f*r0

    Q = c['n']*np.exp(d['n']*PHI['n']**e['n'])
    Q = Q + c['s']*np.exp(d['s']*PHI['s']**e['s'])

    return r, Q


def MP_Shue1998(th, DP, Bz):
    ''' Shue 1998 Magnetopause model. Returns the MP distance for given
    theta (r), dynamic pressure (in nPa) and Bz (in nT).
    * theta: Angle from the x axis (model is cylindrical symmetry)
    * PD: Dynamic Pressure in nPa
    * Bz: z component of IMF in nT'''
    r0 = (10.22+1.29*np.tanh(0.184*(Bz+8.14)))*DP**(-1./6.6)

    a = (0.58-0.007*Bz)*(1+0.024*np.log(DP))
    return r0*(2./(1+np.cos(th)))**a
