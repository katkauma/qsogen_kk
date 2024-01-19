from setuptools import setup, find_packages

setup(
    name='qsogen',
    version='0.1',    
    description='Parametric models for quasar SEDs, including capabilities to produce synthetic photometry and colors. A fork of qsogen by MJTemple that extends the available galaxy templates and updates the IGM absoprtion model.',
    url='https://github.com/katkauma/qsogen_kk',
    license='MIT license',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'astropy',                   
                      ],
    include_package_data=True,
    package_data={'qsogen':['filterinfo.json'],'qsogen.data':['*'],'qsogen.filters':['*.filter'],'qsogen.galaxy_templates':['*.sed']}
)
