"""
Code to process and calculate IGM absorption using different models
"""

import numpy as np
from scipy.special import factorial
from mpmath import gamma as gammafn
from mpmath import gammainc


c = 2.998e5
H0 = 70
OmegaM = 0.3

'''def tau_eff_kauma_highz(z):
    A = 2.13369912
    z_b = 5.17124349
    a_1 = -2.32643103
    a_2 = -7.01644977
    delta = 0.05886805
    c = -0.09805639

    tau_eff = A*(z/z_b)**(-a_1) * (0.5*(1+(z/z_b)**(1/delta)))**((a_1-a_2)*delta)+c
    return np.where(tau_eff < 0, 0., tau_eff)'''

'''def tau_eff_kauma(z):#, A, z_b, z_c, a_1, a_2, a_3, delta_1, delta_2):#,c):
    
    ''''''
    A = 0.04130959
    z_b1 = 1.36430655
    z_b2 = 5.11325679
    a_1 = -0.47148930
    a_2 = -2.83726276
    a_3 = -6.34685654
    delta_1 = 0.16105344
    delta_2 = 0.03990376
    
    return A*((z)/(z_b1))**(-a_1) * ((1.+((z)/(z_b1))**(1./delta_1)))**((a_1-a_2)*delta_1)*((1.+((z)/(z_b2))**(1./delta_2)))**((a_2-a_3)*delta_2) '''
    
def osc_strength_lys(n):
    num = 2.**8. * np.power(n,5.) * np.power(n-1., 2.*n-4.)
    den = 3. * np.power(n+1., 2.*n+4.)
    return num/den

def wavelength_lys(n):
    R_H = 1.09678e-3
    inv_wv = R_H*(1. - n**-2.)
    
    return inv_wv**-1. #in angstroms

def cross_section(n, b=28):
    K = 0.014974365635221684 #sqrt(pi) * e^2 /(m_e *c) (in cgs units)
    b_cm = b*1e5 #convert from km/s to cm/s
    line_wvs_cm = wavelength_lys(n)/1e8 # convert from angstroms to cm 
    
    
    
    return K*osc_strength_lys(n)*line_wvs_cm/b_cm


def double_pl(z, A, z0, g1, g2):
    return A*np.piecewise(
        z,
        [z<z0,z>=z0],
        [lambda z:((1+z)/(1+z0))**g1,
         lambda z:((1+z)/(1+z0))**g2]
        )
    
def triple_pl(z, A=None, z0=None, z1=None, g1=None, g2=None, g3=None):
    return A*np.piecewise(
        z,
        [z<z1,(z>=z0) & (z<z1), z>=z1],
        [lambda z:((1+z)/(1+z0))**g1,
            lambda z:((1+z)/(1+z0))**g2,
            lambda z:((1+z1)/(1+z0))**g2*((1.+z)/(1+z1))**g3]
    )


def ntau_laf_lc_eq(l_obs, z_s, z_b=4.93, A = 23.5857, eta1=4.27, eta2=17.73, alpha=2.75):
    z_L = l_obs/912.-1.
    taus = np.zeros_like(z_L)
    
    
    def int1(zmin, zmax):
        int_factor_1 = (eta1-alpha-1.5)
        #print(zmax)
        if zmax>=z_b:
            zmax = z_b
        #print(zmax)
        
        return np.piecewise(zmin, 
                            [zmin<0., (zmin> 0) & (zmin<zmax), zmin>=zmax],
                            [0., lambda z: ((1+z_b)**-eta1)/int_factor_1 * ((1+zmax)**int_factor_1 - (1+z)**int_factor_1),
                             0.])
    
    def int2(zmin, zmax):
        int_factor_2 = (eta2-alpha-1.5)
        #if zmax<=z_b:
            #zmax = z_b
        return np.piecewise(zmin, 
                            [zmin<z_b, 
                             (zmin>=z_b) & (zmin<zmax), 
                             (zmin>=zmax) | (zmax<=z_b)] ,
                            [((1+z_b)**-eta2) /int_factor_2 * ((1+zmax)**int_factor_2 - (1+z_b)**int_factor_2), 
                             lambda z: ((1+z_b)**-eta2) /int_factor_2 * ((1+zmax)**int_factor_2 - (1+z)**int_factor_2),
                             0.])
    
    taus = int1(z_L, z_s)+int2(z_L, z_s)
    #print(int2(z_L, z_s))
    return  c/(A*H0*np.sqrt(OmegaM)) * (1+z_L)**alpha * taus

def ntau_laf_ls_eq(l_obs, z_s, beta=1.7,nmax=41, Rn=None):


    nrange = np.arange(2,nmax+1)
    
    if Rn is not None:
        Rn = Rn[:,None]
    else:
        Rn = (np.array(cross_section(nrange)/cross_section(2))**(beta-1))[:,None]
    
    linewvs = np.array([1215.67983856, 1025.72986378,  972.54387085,  949.74987387,
        937.81016117,  930.7548764 ,  926.23225795,  923.1568774 ,
        920.96957466,  919.35787791,  918.13582213,  917.18702105,
        916.43557061,  915.83023552,  915.33540786,  914.92571183,
        914.58266492,  914.29254525,  914.0449914 ,  913.83206046,
        913.64758053,  913.48669687,  913.34554827,  913.22103257,
        913.11063429,  913.01229633,  912.92432321,  912.84530735,
        912.77407233,  912.70962879,  912.6511398 ,  912.59789351,
        912.54928141,  912.50478078,  912.4639406 ,  912.42637006,
        912.39172915,  912.35972094,  912.33008522])
    #lya forest, lyman series
    
    if nmax!=len(Rn):
        linewvs = linewvs[:nmax-1]
        Rn = Rn[:nmax-1]

    
    def tau_lya(z, params=None):
        if params is None:
            # params = dict({'A': np.float64(0.1491715088365581),
            #                 'z0': np.float64(2.099795519663383),
            #                 'z1': np.float64(5.224545059579457),
            #                 'g1': np.float64(2.0551398962167933),
            #                 'g2': np.float64(3.7645046991285818),
            #                 'g3': np.float64(7.447695345778327)})
            # params =  dict(A = 0.05038254,
            #         z0 = 1.29999998,
            #         z1 = 5.03328709,
            #         g1 = 1.41248811,
            #         g2 = 3.67142108,
            #         g3 = 6.77804632)
            params = {'A': np.float64(0.05329152207475722),
                        'z0': np.float64(1.310159185590917),
                        'z1': np.float64(5.026197331022725),
                        'g1': np.float64(1.4402790516099375),
                        'g2': np.float64(3.6264760964755287),
                        'g3': np.float64(6.769737774165061)}
        return triple_pl(z,**params)
    
    zlook = np.outer(1./linewvs,l_obs)-1
    #print(zlook)
    tau_laf_i = np.zeros_like(zlook)
    # only compute for l_obs<1216*(1+zs)
    mask_lya = zlook<z_s
    tau_laf_i[mask_lya] = tau_lya(zlook[mask_lya])
    tau_laf_i *= Rn
    
    return np.sum(tau_laf_i, axis=0)

def ntau_lls_lc_eq(l_obs, z_s, 
                  logNb=21., beta1=1.2, beta2=2.6, B=766.2277,
                  l_0=1.46, z_0=3.0, gamma=1.70,
                  alpha=2.75,
                  kmax=10):
    
    sigmaL = 6.3E-18
    A = l_0*(1+z_0)**-gamma
    if B==None:
        B = 10.**((logNb)*(beta1-1)) * ( (10.**((17.5-logNb)*(1.-beta1)) - 1.)/(beta1-1.) + 1./(beta2-1.) )**-1.
    
    beta_m1 = beta1-1.
    gamma_p1 = gamma+1.
    taub = sigmaL*10.**logNb
    zs_p1 = z_s+1.
    zl_p1 = l_obs/912.
    coeff = A*(B)*sigmaL**beta_m1
    zl_p1_gamma_p1 =zl_p1**gamma_p1
    #zl_p1_gamma_p1[zl_p1>=1]=0

    
    
    
    term1 = (1-taub**-beta_m1)/(beta_m1*gamma_p1) * (zs_p1**gamma_p1 - zl_p1_gamma_p1)
    
    term2_coeff =  gamma_p1 - alpha*beta_m1
    term2 = float(gammafn(-beta_m1)) / term2_coeff * (zs_p1**term2_coeff * np.power(zl_p1, alpha*beta_m1) - zl_p1_gamma_p1)
    
    k = np.arange(0, kmax+1, dtype=float)[:,None]
    term3_coeff = gamma_p1 - alpha*k
    term3_sum = (-1)**k/factorial(k) * 1/(beta_m1 - k) * 1/(term3_coeff) * (zs_p1**term3_coeff * zl_p1**(alpha*k) - zl_p1_gamma_p1)
    term3 = np.sum(term3_sum, axis=0)
    
    mask = (zl_p1<zs_p1) & (zl_p1>=1)
    taus = coeff * (term1 - term2 - term3)
    taus[~mask] = 0.
    
    return taus
    
def ntau_lls_ls_eq(l_obs, z_s, 
                logNb=21., beta1=1.2, beta2=2.6, B=None,
                l_0=1.46, z_0=3.0, gamma=1.70,
                alpha=2.75,
                kmax=10,
                nmax=40, 
            ratios=np.array([1., 1., 1., 1., 1.,
                            1.        , 1.        , 1.        , 0.99999994, 0.999998  ,
                            0.99997744, 0.99986838, 0.99950535, 0.99862922, 0.99694284,
                            0.99418282, 0.99016966, 0.98482453, 0.97816014, 0.9702588 ,
                            0.96124781, 0.951278  , 0.94050736, 0.92908997, 0.91716912,
                            0.90487374, 0.89231704, 0.87959651, 0.8667948 , 0.85398111,
                            0.8412126 , 0.82853603, 0.8159891 , 0.80360187, 0.79139792,
                            0.77939538, 0.76760783, 0.7560451 , 0.74471389])):
    
    beta1_m1 = beta1-1.
    beta2_m1 = beta2-1.
    gamma_p1 = gamma+1.
    zs_p1 = z_s+1.
    db_c = 5*28/2.998E5
    
    line_ns = np.arange(2, nmax+1)
    
    linewvs = np.array([1215.67983856, 1025.72986378,  972.54387085,  949.74987387,
        937.81016117,  930.7548764 ,  926.23225795,  923.1568774 ,
        920.96957466,  919.35787791,  918.13582213,  917.18702105,
        916.43557061,  915.83023552,  915.33540786,  914.92571183,
        914.58266492,  914.29254525,  914.0449914 ,  913.83206046,
        913.64758053,  913.48669687,  913.34554827,  913.22103257,
        913.11063429,  913.01229633,  912.92432321,  912.84530735,
        912.77407233,  912.70962879,  912.6511398 ,  912.59789351,
        912.54928141,  912.50478078,  912.4639406 ,  912.42637006,
        912.39172915,  912.35972094,  912.33008522])
        
    if nmax!=len(ratios):
        linewvs = linewvs[:nmax-1]
        ratios = ratios[:nmax-1]
    Nb = 10**logNb
    Nmin = 10**17.2
    if ratios is None:
        sigmas = cross_section(line_ns)
        sigma_alpha = sigmas[0]
        ratios = tau_lls_ls_ratios(Nb=Nb, beta1=beta1, sigma=sigmas)
    else:
        sigma_alpha = 2.70587835e-14

    
    B = 10**(logNb*(beta1_m1)) * ( ((Nmin/Nb)**-beta1_m1-1.)/beta1_m1 + beta2_m1**-1 )**-1
   
    I_alpha = (Nmin**-beta1_m1 - Nb**-beta1_m1)/beta1_m1 - sigma_alpha**beta1_m1 * float(gammainc(-beta1_m1, Nmin*sigma_alpha))
    
    def tau_lls_lya(z):
        return l_0 * (1+z_0)**-gamma *db_c * B * (1+z)**gamma_p1 * I_alpha
    
    # create the taus
    zlook = np.outer(1./linewvs,l_obs)-1
    tau_i = np.zeros_like(zlook)
    mask_lya = zlook<z_s 
    tau_i[mask_lya] = tau_lls_lya(zlook[mask_lya])
    tau_i *= ratios[:,None]
    
    return np.sum(tau_i, axis=0)
    
def ntau_lls_ls_ratios(Nb, beta1, sigma):
    Nmin = 10**17.2
    beta1_m1 = beta1-1.
    frac_coeff = (beta1-1)/(Nmin**-beta1_m1 - Nb**-beta1_m1)
    gams = np.array([float(gammainc(-beta1_m1, Nmin * s)) for s in sigma])
    return 1 - frac_coeff * sigma**beta1_m1 * gams

def newest_trans(obswav, z_s,nmax=30, lc=True, laf_ls_kwargs = {}, laf_lc_kwargs = {}, lls_ls_kwargs = {}, lls_lc_kwargs = {}):
    
    beta_laf = laf_ls_kwargs.get('beta', 1.7)
    Rn_laf = (np.array(cross_section(np.arange(2, nmax+1))/cross_section(2))**(beta_laf-1))
    
    laf_ls = ntau_laf_ls_eq(obswav, z_s, nmax=nmax, Rn=Rn_laf, **laf_ls_kwargs)
    laf_lc = ntau_laf_lc_eq(obswav, z_s, **laf_lc_kwargs)
    
    lls_ls = ntau_lls_ls_eq(obswav, z_s, nmax=nmax, **lls_ls_kwargs)
    lls_lc = ntau_lls_lc_eq(obswav, z_s, **lls_lc_kwargs)
    
    if lc==True:
        return np.exp(-(laf_ls+laf_lc+lls_ls+lls_lc))
    else:
        return np.exp(-(laf_ls+lls_ls))

################################################################################


        
def _lc_igm_part(zi, zf, zb, alpha):
    A = 310
    beta_m1 = 0.7
    const = 0.342
    a_3bm1 = alpha-3*beta_m1
    
    #### note to self - this doesnt work for the z>z2 case, theres an extra factor

    
    return const*(1+zb)**-(alpha-1) * (a_3bm1)**-1 * ((1+zf)**a_3bm1 - (1+zi)**a_3bm1)
    
    
def tau_lc_igm_new(zs, l_obs):
    z_1 = 1.339
    z_2 = 5.269
    a_1 = 1.352
    a_2 = 3.824
    a_3 = 7.599
    # a_1=1.5164
    # a_2=3.6385
    # a_3=6.6101
    # z_1=1.3111
    # z_2=5.0032
    
    lratio= (l_obs/912.)
    
    if zs<z_1:
        return lratio**2.1 * _lc_igm_part(zi=lratio-1, zf=zs, zb=z_1, alpha=a_1)
        # return -0.3394 * lratio**2.1 * ((1+zs)**-0.75 - (lratio)**-0.75)

    elif (zs>=z_1) & (zs<z_2):
        return lratio**2.1 * np.piecewise(lratio,
                                        [lratio<z_1+1, lratio>=z_1+1], 
                                        [lambda lratio: _lc_igm_part(lratio-1, zf=z_1, zb=z_1, alpha=a_1) + _lc_igm_part(zi=z_1, zf=zs, zb=z_1, alpha=a_2),
                                         lambda lratio: _lc_igm_part(lratio-1, zf=zs, zb=z_1, alpha=a_2)])
        
        # return lratio**2.1 * np.piecewise(lratio, 
        #                     [lratio<z_1+1, lratio>=z_1+1], 
        #                     [lambda lratio: 0.018*(1+zs)**1.72 + 0.339*lratio**-0.75 - 0.257,
                             
        #                     lambda lratio: 0.018*((1+zs)**1.72 - lratio**1.72)]
        #                      )

    elif (zs>=z_2):
        return lratio**2.1 * np.piecewise(lratio,
                                        [lratio<z_1+1, (lratio>=z_1+1) & (lratio<z_2+1), lratio>=z_2+1],
                                        [lambda lratio: _lc_igm_part(lratio-1, zf=z_1, zb=z_1, alpha=a_1) + _lc_igm_part(zi=z_1, zf=z_2, zb=z_1, alpha=a_2) + ((1+z_2/1+z_1))**(a_2-1)*_lc_igm_part(zi=z_2, zf=zs, zb=z_2, alpha=a_3),
                                         lambda lratio: _lc_igm_part(zi=lratio-1, zf=z_2, zb=z_1, alpha=a_2) + ((1+z_2)/(1+z_1))**(a_2-1)*_lc_igm_part(zi=z_2, zf=zs, zb=z_2, alpha=a_3),
                                         lambda lratio: ((1+z_2)/(1+z_1))**(a_2-1)*_lc_igm_part(zi=lratio-1, zf=zs, zb=z_2, alpha=a_3)])
        # return lratio**2.1 * np.piecewise(lratio, 
        #                     [lratio<z_1+1, (lratio>=z_1+1) & (lratio<z_2+1), lratio>=z_2+1], 
        #                     [lambda lratio: 5.5e-6*(1+zs)**5.5 + 0.339*lratio**-0.75 - 0.317,
                             
        #                     lambda lratio: 5.5e-6*(1+zs)**5.5 - 0.018*lratio**1.72 + 0.2898,
                             
        #                     lambda lratio: 5.5e-6*((1+zs)**5.5 - lratio**5.5)]
        #                      )
    
    
    
def tau_lc_igm_lls_new(zs, l_obs):
    tau_igm = tau_lc_igm_new(zs, l_obs)
    lratio = l_obs/911.8
    zs_p1 = 1.+zs

    gammafn = 0.3134#0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.15
    gamma = 1.94
    #beta = 1.5
    n = np.arange(9) #first 10 terms cause convergence
    
    term1 = gammafn - np.exp(-1.)
    term2 = np.sum(np.power(-1.,n) / (factorial(n) * (2.*n-1.)))
    term3 = (zs_p1**(-0.5+gamma) * lratio**1.5 - lratio**(gamma+1.))

    term4 = np.sum(np.array([((0.5 * np.power(-1.,n) / (factorial(n) * ((3.*n - gamma - 1) * (n -0.5)))) * (zs_p1**(gamma +1 - (3.*n)) * lratio**(3.*n) - lratio**(gamma + 1))) for n in np.arange(1,10)]), axis=0)

    tau_lls = n0 / (gamma - 0.5) * ((term1 - term2) * term3 - term4)
    return tau_igm+tau_lls

def tau_new_tpl_inoue_dla(zs, l_obs):
    tau_igm = tau_lc_igm_new(zs, l_obs)
    
    lratio = l_obs/911.8
    zs_p1 = 1.+zs
    zlim = 911.8*zs_p1
    if zs<2.0:
        tau_dla=0.211*zs_p1**2.0 - 7.66e-2*zs_p1**2.3*lratio**(-0.3) - 0.135*lratio**2.
    else:
        zs_term1 = zs_p1**3.0
        zs_term2 = zs_p1**3.3
        tau_dla= np.piecewise(lratio, [lratio<3.,lratio>=3.], [lambda lratio: 0.634 + 4.7e-2*zs_term1 - 1.78e-2*zs_term2*lratio**(-0.3) - 0.135*lratio**2. - 0.291*lratio**(-0.3),
        lambda lratio: 4.7e-2*zs_term1 - 1.78e-2*zs_term2*lratio**(-0.3) - 2.92e-2*lratio**3.])
        
    return tau_igm+tau_dla
    
    
    
     
####################

def tau_eff_becker2013(z):
    #Ly alpha optical depth from Becker et al. 2013MNRAS.430.2067B.
    
    tau0=0.3
    beta=13.7
    C=1.35
    z0=4.8

    tau_eff = tau0*((1+z)/(1+z0))**beta+C
    tau_eff = 0.751*((1 + z) / (1 + 3.5))**2.90 - 0.132
    
    return np.where(tau_eff < 0., 0., tau_eff)

def tau_eff_laf_inoue2014(z):
    t = np.piecewise(z,[z<1.2,(z>=1.2) & (z<4.7), z>=4.7], [lambda z:1.690e-2*(1.+z)**1.2, lambda z:2.354e-3*(1.+z)**3.7, lambda z:1.026e-4*(1.+z)**5.5])
    return t

def tau_eff_dla_inoue2014(z):
    t = np.piecewise(z,[z<2.,z>=2.], [lambda z:1.617e-4*(1.+z)**2.0, lambda z:5.390e-5*(1.+z)**3.0])
    return t

def tau_eff_madau1995(z):
    t = 0.0036*(1.+z)**3.46
    return t

def tau_eff_meiksin2006(z):
    t = np.piecewise(z,[z<4.,z>=4.],[lambda z:0.00211*(1.+z)**3.7, lambda z:0.00058*(1.+z)**4.5])
    return t

## lyman continuum absorption
def tau_lc_kauma(zs, l_obs):
    lratio = l_obs/911.8
    zs_p1 = 1.+zs
    gammafn = 0.3134#Gamma(0.72,1) i.e., Gamma(2-beta,1) with beta = 1.28
    #gammafn = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.15 #mistakenly had at 0.25 earlier. values from songaila cowie 2010
    gamma = 1.94
    #beta = 1.5
    n = np.arange(9) #first 10 terms cause convergence
    
    term1 = gammafn - np.exp(-1.)
    term2 = np.sum(np.power(-1.,n) / (factorial(n) * (2.*n-1.)))
    term3 = (zs_p1**(-0.5+gamma) * lratio**1.5 - lratio**(gamma+1.))

    term4 = np.sum(np.array([((0.5 * np.power(-1.,n) / (factorial(n) * ((3.*n - gamma - 1) * (n -0.5)))) * (zs_p1**(gamma +1 - (3.*n)) * lratio**(3.*n) - lratio**(gamma + 1))) for n in np.arange(1,10)]), axis=0)

    tau_lls = n0 / (gamma - 0.5) * ((term1 - term2) * term3 - term4)
    return tau_lls


def tau_lc_inoue(zs,l_obs):
    # only give wavelengths less than 912*(1+zobj)
    # zs is the redshift of the object
    # l_obs is the observed wavelength
    lratio = l_obs/911.8
    zs_p1 = 1.+zs
    zlim = 911.8*zs_p1
    if zs<1.2:
        zs_term = zs_p1**(-0.9)
        tau_laf= 0.325*(lratio**1.2-zs_term*(lratio**2.1))
    elif (zs<4.7) & (zs>=1.2):
        zs_term = zs_p1**1.6
        tau_laf= np.piecewise(lratio, [lratio<2.2,lratio>=2.2], [lambda lratio: 2.55e-2*zs_term * lratio**2.1 + 0.325*lratio**1.2 - 0.25*lratio**2.1, lambda lratio: 2.55e-2 * (zs_term * lratio**2.1 - lratio**3.7)])
        
    else:
        zs_term = zs_p1**3.4
        tau_laf= np.piecewise(lratio,[lratio<2.2,(lratio>=2.2) & (lratio<5.7),lratio>=5.7],
                           [lambda lratio: 5.22e-4*zs_term*lratio**2.1 + 0.325*lratio**1.2 - 3.14e-2*lratio**2.1,
                            lambda lratio: 5.22e-4*zs_term*lratio**2.1 + 0.218*lratio**2.1 - 2.55e-2*lratio**3.7,
                            lambda lratio: 5.22e-4*(zs_term*lratio**2.1 - lratio**5.5)]) # 5.07e-2 works better
    
    #dla
    if zs<2.0:
        tau_dla=0.211*zs_p1**2.0 - 7.66e-2*zs_p1**2.3*lratio**(-0.3) - 0.135*lratio**2.
    else:
        zs_term1 = zs_p1**3.0
        zs_term2 = zs_p1**3.3
        tau_dla= np.piecewise(lratio, [lratio<3.,lratio>=3.], [lambda lratio: 0.634 + 4.7e-2*zs_term1 - 1.78e-2*zs_term2*lratio**(-0.3) - 0.135*lratio**2. - 0.291*lratio**(-0.3),
        lambda lratio: 4.7e-2*zs_term1 - 1.78e-2*zs_term2*lratio**(-0.3) - 2.92e-2*lratio**3.])
    return tau_laf+tau_dla



def tau_lc_madau(zs, l_obs):
    zs_p1 = 1.+zs
    lratio = l_obs/911.8
    tau = 0.25*lratio**3. * (zs_p1**0.46 - lratio**0.46) + 9.4*lratio**1.5 * (zs_p1**0.18 - lratio**0.18) - 0.7*lratio**3. * (lratio**(-1.32) - zs_p1**(-1.32)) - 0.023* (zs_p1**1.68 - lratio**1.68)
    return tau

def tau_lc_meiksin(zs, l_obs):
    lratio = l_obs/911.8
    zs_p1 = 1.+zs
    tau_igm = 0.805*lratio**3. * (1./lratio - 1./zs_p1)

    gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    n = np.arange(9) #first 10 terms cause convergence
    
    term1 = gamma - np.exp(-1.)
    term2 = np.sum(np.power(-1.,n) / (factorial(n) * (2.*n-1.)))
    term3 = (zs_p1 * lratio**1.5 - lratio**2.5)

    term4 = np.sum(np.array([((2. * np.power(-1.,n) / (factorial(n) * ((6.*n - 5.) * (2. *n -1)))) * (zs_p1**(2.5 - (3.*n)) * lratio**(3.*n) - lratio**2.5)) for n in np.arange(1,10)]), axis=0)

    tau_lls = n0 * ((term1 - term2) * term3 - term4)
    return tau_igm+tau_lls



############################################

# return the transmission for each object

def calc_transmission(z,wavred,model, lc=True, nmax=31, **kwargs):
    debug=False
    # assign lines, ratios, etc
    lines, ratios, tau_lya_model = None, None, None
    if debug:
        print('\n'+model)

    if model=='NEW':
        return newest_trans(obswav=wavred, z_s=z,nmax=nmax, lc=lc, **kwargs)
    elif model=='NEW_1p65':
        return newest_trans(obswav=wavred, z_s=z,nmax=nmax, lc=lc, laf_ls_kwargs={'beta':1.65})
        
    else:
        if model=='inoue+2014':
            if debug:
                print('model is', model)
            lines = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226,
                            923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324])
            
            ratios = np.array([1.00000000e+00, 2.77633136e-01, 1.32485207e-01,
                            7.80473373e-02, 5.15207101e-02, 3.65562130e-02, 
                            2.72721893e-02, 2.11183432e-02, 1.68224852e-02, 
                            1.37159763e-02, 1.13786982e-02, 9.59763314e-03, 
                            8.19526627e-03, 7.07692308e-03, 6.17159763e-03, 
                            5.42840237e-03, 4.80946746e-03, 4.29053254e-03, 
                            3.84911243e-03, 3.47218935e-03, 3.14733728e-03, 
                            2.86568047e-03, 2.61952663e-03, 2.40414201e-03, 
                            2.21183432e-03, 2.04378698e-03, 1.89289941e-03, 
                            1.75798817e-03, 1.63668639e-03, 1.52781065e-03, 
                            1.42899408e-03, 1.33905325e-03, 1.25798817e-03, 
                            1.18343195e-03, 1.11538462e-03, 1.05266272e-03, 
                            9.95266272e-04, 9.42603550e-04, 8.93491124e-04])[:,None]

            tau_lya_model = tau_eff_laf_inoue2014
            tau_lc_model = tau_lc_inoue
                

        
        elif model=='becker+2013':
            if debug:
                print('model is becker+2013')
            lines = np.array([1215.67, 1025.72, 972.537])
            ratios = np.array([1.,0.19005811214447021,0.06965703475200001])[:,None]
            tau_lya_model = tau_eff_becker2013
            lc=False
            
        elif model=='madau1995':
            if debug:
                print('model is madau1995')
            lines = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803,930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329,914.919, 914.576])
            ratios = (np.array([0.0036,0.0017,0.0011846,0.0009410,0.0007960,0.0006967,0.0006236,0.0005665,0.0005200,0.0004817,0.0004487,0.0004200,0.0003947,0.000372,0.000352,0.0003334,0.00031644])/0.0036)[:,None]
            tau_lya_model = tau_eff_madau1995
            tau_lc_model = tau_lc_madau
        
        elif model=='meiksin2006':
            if debug:
                print('model is meiksin 2006')
            lines = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, # n=2 to n=9
                            920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703]) #up to n=31
            
            ratios = np.array([1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283, # n=2 to n=9
                            0.02058182, 0.01543636, 0.01187413, 0.00932967,
                            0.00746374, 0.00606429, 0.00499412, 0.00416176, 0.00350464, # n=10-31; t/tau_alpha = 20.376/(n(n^2-1))
                            0.00297895, 0.00255338, 0.00220519, 0.00191756, 0.00167787,
                            0.00147652, 0.00130615, 0.00116103, 0.00103663, 0.00092939,
                            0.00083645, 0.00075551, 0.00068468])
            
            if z<3.:
                ratios[1:] *= (0.25*(1.+z))**(1./3.)
            elif z>=3.:
                ratios[1:4] *= (0.25*(1.+z))**(1./6.)
                #ratios[1:4] += (0.25*(1.+z)**(1./6.))
                ratios[5:] *= (0.25*(1.+z))**(1./3.)

                
            ratios = ratios[:,None]
            
            tau_lya_model = tau_eff_meiksin2006
            tau_lc_model = tau_lc_meiksin
            
        else:
            raise ValueError("model must be 'kauma+', 'inoue+2014', 'becker+2013', 'meiksin2006' or 'madau1995'.")

            
        zlook = np.outer(1./lines,wavred)-1.
        tau_laf_i = np.zeros_like(zlook)
        mask = zlook<z
        tau_laf_i[mask] = tau_lya_model(zlook[mask])
        tau_laf_i *= ratios
        tau_laf = np.sum(tau_laf_i,axis=0)

        if model not in ['inoue+2014', 'tpl_pow_p_inoue']:
            tau_dla = 0.
        else:
            dla_ratios = np.array([1., 0.9554731 , 0.92640693,
                0.90290662, 0.88373531, 0.86703772, 
                0.85157699, 0.83797155, 0.82560297, 
                0.81385281, 0.80272109, 0.79220779, 
                0.78231293, 0.77303649, 0.76437848, 
                0.75572047, 0.74768089, 0.74025974,
                0.73283859, 0.72541744, 0.71861472, 
                0.711812  , 0.70500928, 0.69882498, 
                0.69264069, 0.6864564 , 0.68089054, 
                0.67470625, 0.66914038, 0.66357452, 
                0.65862709, 0.65306122, 0.64811379, 
                0.64316636, 0.63821892, 0.63327149, 
                0.62894249, 0.62399505, 0.61966605])[:,None]
            tau_dla_i = np.zeros_like(zlook)
            tau_dla_i[mask] = tau_eff_dla_inoue2014(zlook[mask])
            tau_dla = np.sum(tau_dla_i*dla_ratios,axis=0)
        
        if lc==True:
            #tau_lc_model = tau_lc_inoue
            tau_lc = np.zeros_like(wavred)
            mask = wavred<911.8*(1.+z)
            tau_lc[mask] = tau_lc_model(z,wavred[mask])
        else:
            tau_lc= 0.0

        #calculate transmission
        
        return np.exp(-(tau_laf+tau_dla+tau_lc))


    
