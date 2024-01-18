import numpy as np
from scipy.integrate import simps
import os
import json
import pprint
import glob

install_path = os.path.dirname(os.path.abspath(__file__))



files = glob.glob(install_path+'/filters/*.filter')
flist = [item.split('/')[-1][:-7] for item in files]


def produce_zeropoints(system='Vega',
                       filters=flist):
    """Produce the Vega and AB zero points for filters listed in all_filters variable.
    Zero points are pre-computed to save time.
    If you want to compute model photometry in additional filters, first use
    this function to compute their zeropoints and add to dictionaries above.
    """

    waves, responses = [], []
    for band in filters:
        waves.append(wavarrs[band])
        responses.append(resparrs[band])
    #print(system + '_zeropoints = dict(')
    zp_dict = {}

    if system == 'Vega':
        wav_Vega, flux_Vega = np.genfromtxt('data/vega_2007.lis', unpack=True)
        # Vega spectrum
        fluxes = [np.interp(wav, wav_Vega, flux_Vega) for wav in waves]

        for i in range(len(filters)):
            F = simps(waves[i]*responses[i]*fluxes[i], waves[i])
            zp_dict[filters[i]+'_Vega']=float('{:.6e}'.format(F))
            #print('    ' + filters[i] + '_Vega={:.6e},'.format(F))

    elif system == 'AB':
        const = 0.1088544752  # 3631Jy in erg/s/cm2/A
        # AB system has constant f_nu, so convert to f_lambda
        for i in range(len(filters)):
            F = const*simps(waves[i]**(-1)*responses[i], waves[i])
            zp_dict[filters[i]+'_AB']=float('{:.6e}'.format(F))
            #print('    ' + filters[i] + '_AB={:.6e},'.format(F))
    else:
        raise Exception('System must be "Vega" or "AB"')

    return zp_dict
       
def produce_pivotwv(filters=flist):
    """Produce the inverse squared pivot wavelength for each filter (lambda_pivot**-2). Used in converting between fnu and flambda in get_mags. 
    """

    waves, responses = [], []
    for band in filters:
        waves.append(wavarrs[band])
        responses.append(resparrs[band])
    #print('pivotwv = dict(')
    pivot_dict ={}

    for i in range(len(filters)):
        F = np.sqrt(simps(waves[i]*responses[i], waves[i])/simps(responses[i]*waves[i]**-1,waves[i]))
        pivot_dict[filters[i]]=float('{:.7g}'.format(F))
        
        
        #print('    ' + filters[i] + '={:.7g},'.format(F))

    return pivot_dict
    #print(')')

def vega2ab(vega_zp,ab_zp,filters=flist):
    vega2ab_dict = {}
    for band in filters:
        vega2ab_dict[band]= float(round(-2.5*np.log10(vega_zp[band+'_Vega']/ab_zp[band+'_AB']),4))
    return vega2ab_dict


if __name__ == '__main__':
    #check files
    files = glob.glob(install_path+'/filters/*.filter')
    flist = [item.split('/')[-1][:-7] for item in files]
    
    
    with open(install_path+'/filterinfo.json','r') as file:
        filterinfo = json.load(file)
    newfilters = list(set(flist)-set(filterinfo['Pivot_wv'].keys()))
    missingfilters = list(set(filterinfo['Pivot_wv'].keys())-set(flist)) # filters in filterinfo.json that do not have response curves
    
    filterlist = flist
    

    wavarrs, resparrs = dict(), dict()
    for band in flist:
        try:
            wavarr, response = np.genfromtxt(install_path+'/filters/'+band+'.filter', unpack=True)
        except OSError:
            wavarr, response = np.genfromtxt(install_path+'/filters/'+band+'.filter', unpack=True)
        wavarrs[band] = wavarr
        resparrs[band] = response

    #read in the existing filterinfo.py file

    # get new files
    
    
    vega = produce_zeropoints('Vega',filters=filterlist)
    ab = produce_zeropoints('AB',filters=filterlist)
    pivot = produce_pivotwv(filters=filterlist)
    vega2ab = vega2ab(vega_zp=vega,ab_zp=ab,filters=filterlist)
    
    #merge with the existing list
    filterinfo['Vega_zeropoints'].update(vega)
    filterinfo['AB_zeropoints'].update(ab)
    filterinfo['Pivot_wv'].update(pivot)
    filterinfo['Vega_2_AB'].update(vega2ab)
    
    for band in missingfilters:
        del filterinfo['Vega_zeropoints'][band+'_Vega']
        del filterinfo['AB_zeropoints'][band+'_AB']
        del filterinfo['Pivot_wv'][band]
        del filterinfo['Vega_2_AB'][band]
        
                  
    #print('Information in filterinfo.json: \n')
    #pprint.pprint(filterinfo,indent=4)
    print('Updated or added the following filters in filterinfo.json:')
    for band in filterlist:
        print('\t'+band)
    
    print('\nRemoved the following filters with no transmission file from filterinfo.json:')
    for band in missingfilters:
        print('\t'+band)
    print('\n')
    with open("filterinfo.json","w") as file:
        json.dump(filterinfo,file,indent=4,sort_keys=True)
    


