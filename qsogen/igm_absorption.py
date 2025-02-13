"""
Code to process and calculate IGM absorption using different models
"""

import numpy as np
from scipy.special import factorial


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

def tau_eff_kauma(z):
    z_1= 1.26
    a_1= np.float64(1.43)
    A1= np.float64(0.015531862232038313)
    z_2= 5.14
    a_2= 3.605
    a_3= 7.628
    delta= 0.0295
    A2=1.9866

    return np.piecewise(z,[z<z_1, z>=z_1], [lambda x:A1*((1.+x))**a_1, lambda z:A2*((1.+z)/(1.+z_2))**(a_2)*(0.5*(1.+((1.+z)/(1.+z_2))**(1./delta)))**(delta*(a_3-a_2))])

def tau_eff_kauma_high(z):
    z_2= 5.14
    a_2= 3.605
    a_3= 7.628
    delta= 0.0295
    A2=1.9866
    return A2*((1.+z)/(1.+z_2))**(a_2)*(0.5*(1.+((1.+z)/(1.+z_2))**(1./delta)))**(delta*(a_3-a_2))

    
    
     

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
    gammafn = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    gamma = 1.94
    #beta = 1.5
    n = np.arange(9) #first 10 terms cause convergence
    
    term1 = gammafn - np.exp(-1.)
    term2 = np.sum(np.power(-1.,n) / (factorial(n) * (2.*n-1.)))
    term3 = (zs_p1**(-0.5+gamma) * lratio**1.5 - lratio**(gamma+1.))

    term4 = np.sum(np.array([((0.5 * np.power(-1.,n) / (factorial(n) * ((3.*n - gamma - 1) * (n -0.5)))) * (zs_p1**(gamma +1 - (3.*n)) * lratio**(3.*n) - lratio**(gamma + 1))) for n in np.arange(1,10)]), axis=0)

    tau_lls = n0 / (gamma - 0.5) * ((term1 - term2) * term3 - term4)
    return tau_lls


def tau_lc_kaumaplus(zs, l_obs):
    lratio = l_obs/911.8
    zs_p1 = 1.+zs
    tau_igm = 0.805*lratio**3. * (1./lratio - 1./zs_p1)

    gammafn = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    gamma = 1.94
    #beta = 1.5
    n = np.arange(9) #first 10 terms cause convergence
    
    term1 = gammafn - np.exp(-1.)
    term2 = np.sum(np.power(-1.,n) / (factorial(n) * (2.*n-1.)))
    term3 = (zs_p1**(-0.5+gamma) * lratio**1.5 - lratio**(gamma+1.))

    term4 = np.sum(np.array([((0.5 * np.power(-1.,n) / (factorial(n) * ((3.*n - gamma - 1) * (n -0.5)))) * (zs_p1**(gamma +1 - (3.*n)) * lratio**(3.*n) - lratio**(gamma + 1))) for n in np.arange(1,10)]), axis=0)

    tau_lls = n0 / (gamma - 0.5) * ((term1 - term2) * term3 - term4)
    
    return tau_igm+tau_lls

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
    elif zs<4.7:
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

'''def tau_lc_dla_inoue(zs,lobs):
    lratio = l_obs/911.8
    zs_p1 = 1.+zs
    zlim = 911.8*zs_p1
    if zs<2.0:
        return 0.211*zs_p1**2.0 - 7.66e-2*zs_p1**2.3*lratio**(-0.3) - 0.135*lratio**2.
    else:
        zs_term1 = zs_p1**3.0
        zs_term2 = zs_p1**3.3
        return np.piecewise(lratio, [lratio<3.,lratio>=3.], [lambda lratio: 0.634 + 4.7e-2*zs_term1 - 1.78e-2*zs_term2*lratio**(-0.3) - 0.135*lratio**2. - 0.291*lratio**(-0.3),
        lambda lratio: 4.7e-2*zs_term1 - 1.78e-2*zs_term2*lratio**(-0.3) - 2.92e-2*lratio**3.])'''

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

def calc_transmission(z,wavred,model,lc=True):
    # assign lines, ratios, etc
    lines, ratios, tau_lya_model = None, None, None
    
    if model=='kauma+' or model =='inoue+2014' or model=='kaumaplus':
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
        if model=='kauma+':
            tau_lya_model = tau_eff_kauma
            tau_lc_model = tau_lc_kauma
        elif model=='kaumaplus':
            tau_lya_model = tau_eff_kauma
            tau_lc_model = tau_lc_kaumaplus
        else:
            tau_lya_model = tau_eff_laf_inoue2014
            tau_lc_model = tau_lc_inoue
            

    
    elif model=='becker+2013':
        lines = np.array([1215.67, 1025.72, 972.537])
        ratios = np.array([1.,0.19005811214447021,0.06965703475200001])[:,None]
        tau_lya_model = tau_eff_becker2013
        
    elif model=='madau1995':
        lines = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803,930.748, 926.226, 923.150, 920.963, 919.352])#,918.129, 917.181, 916.429, 915.824, 915.329,914.919, 914.576])
        ratios = (np.array([0.0036,0.0017,0.0011846,0.0009410,0.0007960,0.0006967,0.0006236,0.0005665,0.0005200,0.0004817])/0.0036)[:,None]#,0.0004487,0.0004200,0.0003947,0.000372,0.000352,0.0003334,0.00031644])/0.0036)[:,None]
        tau_lya_model = tau_eff_madau1995
        tau_lc_model = tau_lc_madau
    
    elif model=='meiksin2006':
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

    if model != 'inoue+2014':
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


    
