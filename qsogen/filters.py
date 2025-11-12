import numpy as np
from scipy.integrate import simpson as simps
import os
import json
import glob
import urllib.request
import io
from datetime import date
today = date.today()


install_path = os.path.dirname(os.path.abspath(__file__))

filterdict_file = install_path+'/filterinfo.json'
filter_directory = install_path+'/filter_data/'

with open(filterdict_file,'r') as file:
    filterinfo = json.load(file)

wav_Vega, flux_Vega = np.genfromtxt(f'{filter_directory}vega_2007.lis', unpack=True)

    
def get_filter(name, norm = False):
    #return filter response curve
    wav, flux = np.genfromtxt(f'{filter_directory}{name}.filter', unpack=True)
    if norm:
        flux = flux/np.max(flux)
    return wav, flux

def produce_zeropoints(wave, response, system='Vega', return_vega2ab=True):
    """
    If you want to compute model photometry in additional filters, first use
    this function to compute their zeropoints.  
    """
    if system == 'Vega':
        
        # Vega spectrum
        flux = np.interp(wave, wav_Vega, flux_Vega)
        F = simps(wave*response*flux, x=wave)
        zp=float('{:.6e}'.format(F))
        
        return zp

    elif system == 'AB':
        const = 0.1088544752  # 3631Jy in erg/s/cm2/A
        # AB system has constant f_nu, so convert to f_lambda
        F = const*simps(wave**(-1)*response, x=wave)
        zp=float('{:.6e}'.format(F))
        
        return zp
    
    elif system =='both':
        flux = np.interp(wave, wav_Vega, flux_Vega)
        F = simps(wave*response*flux, x=wave)
        vega_zp=float('{:.6e}'.format(F))
        
        const = 0.1088544752  # 3631Jy in erg/s/cm2/A
        # AB system has constant f_nu, so convert to f_lambda
        F = const*simps(wave**(-1)*response, x=wave)
        AB_zp=float('{:.6e}'.format(F))
        
        if return_vega2ab==True:
            vega2ab = -2.5*np.log10(vega_zp/AB_zp)
            return vega_zp, AB_zp, vega2ab
        else:
            return vega_zp, AB_zp
    else:
        raise Exception('System must be "Vega","AB", or "both"')

def produce_pivotwv(wave, response):
    """Produce the inverse squared pivot wavelength for each filter (lambda_pivot**-2). Used in converting between fnu and flambda in get_mags. 
    """

    F = np.sqrt(simps(wave*response,x=wave)/simps(response/wave,x=wave))
    pivot=float('{:.7g}'.format(F))
        
    return pivot
       
def produce_vega2ab(wave, response):
    # if wave and response are None, check filterinfo.json for existing
    
    vega_zp = produce_zeropoints(wave, response, system='Vega')
    AB_zp = produce_zeropoints(wave, response, system='AB')
    vega2ab= -2.5*np.log10(vega_zp/AB_zp)
    return vega2ab

def produce_filterinfo(wave, response):
    vega, ab, vega2ab = produce_zeropoints(wave, response, system='both', return_vega2ab = True)
    pivot = produce_pivotwv(wave, response)

    
    return dict(Vega_zeropoints=vega, AB_zeropoints=ab, Vega2AB=vega2ab, Pivot_wv=pivot)

def add_filter(name, wave, response, overwrite=False, header=''):
        
    if name in filterinfo['Vega_zeropoints'].keys() and overwrite==False:
        raise KeyError(f"{name} is already in filterinfo dict and overwrite==False. Please change overwrite==True if you want to replace the filter")
    elif name in filterinfo['Vega_zeropoints'].keys() and overwrite==False:
        pass
        
    
    header_txt = f'\n{name} filter response curve, added by user on {today}.\n Notes: {header} \nColumns: Wavelength (\AA), Transmission'
    # first, save the filter array in the filter_data folder
    np.savetxt(f'{filter_directory}{name}.filter',np.c_[wave,response], header=header_txt)
    print(f"Filter response curve saved to {install_path}{name}.filter")
    
    # calculate information
    values = produce_filterinfo(wave, response)
    filterinfo['Vega_zeropoints'][name]=values['Vega_zeropoints']
    filterinfo['AB_zeropoints'][name]=values['AB_zeropoints']
    filterinfo['Pivot_wv'][name]=values['Pivot_wv']
    filterinfo['Vega_2_AB'][name]=values['Vega_2_AB']
    
    with open("filterinfo.json","w") as file:
        json.dumps(filterinfo,file,indent=4,sort_keys=True)
    print(f"Updated filterinfo.json with {name}.")
        
def add_filter_from_url(name, url, overwrite=True, header=''):
        
    with urllib.request.urlopen(url) as r:
        data = r.read().decode('utf-8')
    data_io = io.StringIO(data)
    
    wave, response = np.genfromtxt(data_io, unpack=True)
    
    add_filter(name, wave, response, overwrite=overwrite, header=f'Source URL: {url}. {header}')
    
def update_filterinfo(names,waves,responses):

    for name, wave, response in names, waves, responses:
        # calculate information
        vega, ab, vega2ab, pivot = produce_filterinfo(wave, response)
        
        
        filterinfo['Vega_zeropoints'][name]=vega
        filterinfo['AB_zeropoints'][name]=ab
        filterinfo['Pivot_wv'][name]=pivot
        filterinfo['Vega_2_AB'][name]=vega2ab
    
    with open("filterinfo.json","w") as file:
        json.dump(filterinfo,file,indent=4,sort_keys=True)
        
    print(f"Updated filterinfo.json with {names}.") 
    
def add_filter_from_file(name, file, overwrite=True, header=''):
        
    wave, response = np.genfromtxt(file, unpack=True)
    add_filter(name, wave, response, overwrite=overwrite, header=f'Original file: {file}. {header}')
    

def remove_filter(name):
    del filterinfo['Vega_zeropoints'][name]
    del filterinfo['AB_zeropoints'][name]
    del filterinfo['Pivot_wv'][name]
    del filterinfo['Vega_2_AB'][name]
    
    with open("filterinfo.json","w") as file:
        json.dump(filterinfo,file,indent=4,sort_keys=True)
    
    print(f'Deleted {name} entries from filterinfo.json')
    
    try:
        os.remove(f'{filter_directory}{name}.filter')
        print(f'Deleted {filter_directory}{name}.filter')
    except FileNotFoundError as e:
        print(f"'{filter_directory}{name}.filter' does not exist. Did you delete it already?")

    
def remove_filters(names):
    for name in names:
        filterinfo = remove_filters(name)



if __name__ == '__main__':
    #check files
    files = glob.glob(install_path+'/filters/*.filter')
    flist = [item.split('/')[-1][:-7] for item in files]
    
    with open(filterdict_file,'r') as file:
        filterinfo = json.load(file)

    newfilters = list(set(flist)-set(filterinfo['Pivot_wv'].keys()))
    missingfilters = list(set(filterinfo['Pivot_wv'].keys())-set(flist)) # filters in filterinfo.json that do not have response curves
    
    filterlist = flist
    

    wavarrs, resparrs, bands = [], [], []
    for band in flist:
        try:
            wavarr, response = np.genfromtxt(install_path+'/filters/'+band+'.filter', unpack=True)
        except OSError:
            wavarr, response = np.genfromtxt(install_path+'/filters/'+band+'.filter', unpack=True)
        wavarrs.append(wavarr)
        resparrs.append(response)
        bands.append(band)

    #read in the existing filterinfo.py file

    # get new files
    
    for band, wave, response in bands, wavarrs, resparrs:
        vega, ab, vega2ab, pivot = produce_filterinfo(wave, response)
        
        
    
    vega = produce_zeropoints('Vega',filters=filterlist)
    ab = produce_zeropoints('AB',filters=filterlist)
    pivot = produce_pivotwv(filters=filterlist)
    vega2ab = produce_vega2ab(vega_zp=vega,ab_zp=ab,filters=filterlist)
    
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
        

    print('Updated or added the following filters in filterinfo.json:')
    for band in newfilters:
        print('\t'+band)
    
    print('\nRemoved the following filters with no transmission file from filterinfo.json:')
    for band in missingfilters:
        print('\t'+band)
    print('\n')
    with open("filterinfo.json","w") as file:
        json.dump(filterinfo,file,indent=4,sort_keys=True)

