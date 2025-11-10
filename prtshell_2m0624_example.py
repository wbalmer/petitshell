#!/usr/bin/env python
# coding: utf-8

# low level imports
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1" # stop e.g. numpy from doing own parallel with mpi
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # keep hdf5 file locking up across processes 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL) # for rockfish dlopen error

# basic imports
import time
import copy
import numpy as np
from astropy.io import fits
import multiprocess as mp
import warnings
import corner
import astropy.units as u
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
from spectres import spectres
from species import SpeciesInit
from species.phot.syn_phot import SyntheticPhotometry

# sampler imports 
from nautilus import Prior
from nautilus import Sampler

# mpi imports
from mpi4py.futures import MPIPoolExecutor
from schwimmbad import MPIPool
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# prt specific imports
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global, dtdp_temperature_profile
from petitRADTRANS.radtrans import Radtrans # <--- this is our "spectrum generator," e.g. the radiative transfer solver
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.clouds import return_cloud_mass_fraction, simple_cdf
from petitRADTRANS.math import filter_spectrum_with_spline


# general setup
retrieval_name = '2M0624_TEST'
output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
discard_exploration = False
f_live = 0.05
resume = False

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # load data, start species if needed for photometry
    # time.sleep(rank) # sleeping here to avoid two processes trying to write to species.ini at the same time
    # SpeciesInit()

    data_path = './example_data/'
    data = np.loadtxt(data_path+'2M0624_bin300.txt')
    
    w = data[:,0]
    f = data[:,1]
    fe = data[:,2]

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron) # units of pRT models to units of data

    default_params = {
        'R_pl':1.0,
        'plx':82.0248,
        'logg':4.5,

        'T_bottom':5000.,
        'N_layers':10,
        'dPT_10':0.05,
        'dPT_9':0.05,
        'dPT_8':0.05,
        'dPT_7':0.06,
        'dPT_6':0.08,
        'dPT_5':0.16,
        'dPT_4':0.21,
        'dPT_3':0.18,
        'dPT_2':0.15,
        'dPT_1':0.15,

        'C/O':0.55,
        'Fe/H':0.0,
        'log_pquench':0.0,
        'fsed':4,
        'sigma_lnorm':1.5,
        'logKzz':10,
    }

    def likelihood(param_dict, debug=False):

        ln = 0

        params = default_params
        for param in param_dict:
            params[param] = param_dict[param]

        w_i, f_i = spectrum_generator(params)

        # compute sphere likelihood:

        rb_f_i = spectres(w, w_i[0], f_i[0])
        
        if debug:
            plt.figure()
            plt.plot(w, rb_f_i)

        if 'corr_len' in param_dict.keys():
            # from Wang et al. 2020, species.fit.fit_model
            wavel_j, wavel_i = np.meshgrid(w, w)

            error_j, error_i = np.meshgrid(fe, fe)

            corr_len = 10.0 ** param_dict["corr_len"]  # (um)
            corr_amp = param_dict["corr_amp"]

            cov_matrix = (
                corr_amp**2
                * error_i
                * error_j
                * np.exp(-((wavel_i - wavel_j) ** 2) / (2.0 * corr_len**2))
                + (1.0 - corr_amp**2) * np.eye(w.shape[0]) * error_i**2
            )

            ln_i = (
                (f - rb_f_i)
                @ np.linalg.inv(cov_matrix)
                @ (f - rb_f_i)
            )

            ln_i += np.nansum(
                np.log(2.0 * np.pi * fe**2)
            )

            ln_i *= -0.5

            ln += ln_i
        
        else:
            chi2 = np.nansum(((f - rb_f_i)/fe)**2)
            ln += -chi2/2 - np.nansum(np.log(2*np.pi*fe**2)/2)
            
        if debug:
            plt.errorbar(w_i[2], f_i[2], marker='s', color='red')
            plt.savefig('temp_spec.png')
            print(ln)

        return ln


    chem = PreCalculatedEquilibriumChemistryTable()
    # Load scattering version of pRT

    rtpressures = np.logspace(-6, 3, 100) # set pressure range
    line_species = [
        'H2O',
        'CO-NatAbund',
        # '12CO',
        # '13CO',
        'CH4',
        'CO2',
        'HCN',
        'FeH',
        'H2S',
        'NH3',
        'PH3',
        'Na',
        'K',
        'TiO',
        'VO',
        'SiO'
    ]
    rayleigh_species = ['H2', 'He'] # why is the sky blue?
    gas_continuum_contributors = ['H2--H2', 'H2--He'] # these are important sources of opacity
    cloud_species = ['MgSiO3(s)_crystalline__DHS',
                    'Fe(s)_crystalline__DHS'] # these will be important for clouds

    smresl = '1000' # model resolution, R=1000 c-k
    atmosphere = Radtrans(
        pressures = rtpressures,
        line_species = [i+f'.R{smresl}' for i in line_species],
        rayleigh_species = rayleigh_species, # why is the sky blue?
        gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
        cloud_species = cloud_species, # these will be important for clouds
        wavelength_boundaries = [w[0]-0.1, w[-1]+0.1],
        line_opacity_mode='c-k' # lbl or c-k
    )


    def spectrum_generator(params):
        planet_radius = params['R_pl']* cst.r_jup_mean
        parallax = params['plx']
        r2d2 = (planet_radius/(cst.pc/(parallax/1000)))**2
        if 'mass' in params.keys():
            reference_gravity = (cst.G*params['mass']*cst.m_jup)/(planet_radius**2)
        else:
            reference_gravity = 1e1**params['logg']
        
        pressures = atmosphere.pressures * 1e-6 # cgs to bar

        # gradient P-T profile from Zhang+23
        t_bottom = params['T_bottom']
        num_layer = params['N_layers']
        layer_pt_slopes = np.ones(num_layer) * np.nan
        for index in range(num_layer):
            layer_pt_slopes[index] = params[f'dPT_{num_layer - index}']
        temperature = dtdp_temperature_profile(
            pressures,
            num_layer,
            layer_pt_slopes,
            t_bottom,
            top_of_atmosphere_pressure=-6,
            bottom_of_atmosphere_pressure=3
        )

        co_ratio = params['C/O']
        feh = params['Fe/H']
        log_pquench = params['log_pquench']

        co_ratios = co_ratio * np.ones_like(pressures)
        log10_metallicities = feh * np.ones_like(pressures)

        mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
            co_ratios=co_ratios,
            log10_metallicities=log10_metallicities,
            temperatures=temperature,
            pressures=pressures,
            carbon_pressure_quench=10**log_pquench,
            full=True
        )

        if '13CO' in atmosphere.line_species:
            c_iso_ratio = params['C_iso']
            mass_fractions['13CO'] = mass_fractions['CO']/c_iso_ratio
            mass_fractions['12CO'] = mass_fractions['CO']-mass_fractions['13CO']

        sigma_lnorm = params['sigma_lnorm']
        logkzz = params['logKzz']

        # mmw = params['mmw'] # we get mean_molar_masses from chem.interpolate instead of setting it ourselves
        # mean_molar_masses = mmw * np.ones_like(temperature)
        mmw = np.mean(mean_molar_masses)

        eddy_diffusion_coefficients = np.ones_like(temperature)*1e1**logkzz
        if 'fsed' not in params.keys():
            cloud_f_sed = {}
            for specie in atmosphere.cloud_species:
                cloud_f_sed[specie] = params[f'fsed_{specie}']
        else:
            fsed = params['fsed'] # global fsed for now
            cloud_f_sed = {specie:fsed for specie in atmosphere.cloud_species}
        cloud_particle_radius_distribution_std = sigma_lnorm

        cbases = {}
        for specie in atmosphere.cloud_species:
            easy_chem_name = specie.split('_')[0].split('-')[0].split(".")[0]
            cmf = return_cloud_mass_fraction(specie, feh, co_ratio)
            cbase = simple_cdf(specie, pressures, temperature, feh, co_ratio, mmw=mmw)
            cbases[easy_chem_name] = cbase
            mass_fractions_cloud = np.zeros_like(temperature)
            mass_fractions_cloud[pressures<=cbase] = cmf * (pressures[pressures<=cbase] / cbase) ** cloud_f_sed[specie]
            
            if "eq_scaling_" + specie in params.keys():
                mass_fractions_cloud *= (10 ** params['eq_scaling_' + specie]) # Scaled by a constant factor
                
            mass_fractions[specie] = mass_fractions_cloud
            
        for species in line_species:
            easy_chem_name = species.split('_')[0].split('-')[0].split(".")[0]
            if 'FeH' in species:
                # Magic factor for FeH opacity - off by factor of 2
                abunds_change_rainout = copy.copy(mass_fractions[species] / 2.)
                if 'Fe(s)' in cbases.keys():
                    index_ro = pressures < cbases['Fe(s)']  # Must have iron cloud
                    abunds_change_rainout[index_ro] = 0.
                mass_fractions[species] = abunds_change_rainout
                
        
        # set up resolution specific mass fraction dictionaries
        mfs = copy.copy(mass_fractions)
        for key in line_species:
            mfs[key+f'.R{smresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            mfs.pop(key.split('_')[0].split('-')[0].split(".")[0])

        # set up resolution and wavelength range specific R-T calcs for each spectrum/dataset
        wavelengths, flux, additional_returned_quantities = atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mfs,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_f_sed=cloud_f_sed,
            cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
            return_contribution=False
        )

        wavelengths = [wavelengths*1e4, ]
        flux = [flux * r2d2 * unit_conv, ]

        # wavelengths *= 1e4 # to micron

        # if 'rv' in params.keys():
        #     # apply RV shift
        #     radial_velocity = params['rv'] * 1e5
        #     # rv in km/s -> 1e5 to cm/s, cst.c in cm/s, wlen first in cm -> micron by 1e4
            
        #     wavelengths *= np.sqrt((1 + radial_velocity/cst.c)/(1- radial_velocity/cst.c))

        # return wavelengths, flux * r2d2 * unit_conv
        return wavelengths, flux

    if rank==0:
        t_start = time.time()
        test_w, test_f = spectrum_generator(default_params)
        t_end = time.time()
        print('First spectrum Generation time: {:.1f}s'.format(t_end - t_start))
        t_start = time.time()
        ln = likelihood(default_params)
        t_end = time.time()
        print('Likelihood time: {:.1f}s'.format(t_end - t_start))
        s_test_f = spectres(w, test_w[0], test_f[0])
        plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
        plt.plot(w, s_test_f, label=ln, color='red')
        plt.legend()
        plt.savefig(output_dir+'test_alldata_generation.png')

    prior = Prior()
    
    prior.add_parameter('R_pl', dist=truncnorm(0.5, 2.0, loc=1.0, scale=0.1))
    # prior.add_parameter('mass', dist=truncnorm(1e-1, 1e2, 3.75, scale=0.5))
    prior.add_parameter('logg', dist=norm(loc=3.7, scale=0.1))
    
    prior.add_parameter('plx', dist=norm(loc=82.0248, scale=0.3583))

    # prior.add_parameter('T_int', dist=(500, 1500))
    # prior.add_parameter('kappa_ir', dist=(1e-3, 5e-1))
    # prior.add_parameter('gamma', dist=(1e-3, 0.999))

    prior.add_parameter('T_bottom', dist=(2500, 25000))
    # z23, combo of diff. grids from 10^-3 to 10^3
    # prior.add_parameter('dPT_1', dist=norm(loc=0.25, scale=0.025))
    # prior.add_parameter('dPT_2', dist=norm(loc=0.25, scale=0.045))
    # prior.add_parameter('dPT_3', dist=norm(loc=0.26, scale=0.05))
    # prior.add_parameter('dPT_4', dist=norm(loc=0.2, scale=0.05))
    # prior.add_parameter('dPT_5', dist=norm(loc=0.12, scale=0.045))
    # prior.add_parameter('dPT_6', dist=norm(loc=0.07, scale=0.07))

    # z25, sonora diamondback for 2m1207b
    prior.add_parameter('dPT_1', dist=(0.05, 0.25))
    prior.add_parameter('dPT_2', dist=norm(loc=0.15, scale=0.01))
    prior.add_parameter('dPT_3', dist=norm(loc=0.18, scale=0.04))
    prior.add_parameter('dPT_4', dist=norm(loc=0.21, scale=0.05))
    prior.add_parameter('dPT_5', dist=norm(loc=0.16, scale=0.06))
    prior.add_parameter('dPT_6', dist=norm(loc=0.08, scale=0.025))
    prior.add_parameter('dPT_7', dist=norm(loc=0.06, scale=0.02))
    prior.add_parameter('dPT_8', dist=(-0.05, 0.1))
    prior.add_parameter('dPT_9', dist=(-0.05, 0.1))
    prior.add_parameter('dPT_10', dist=(-0.05, 0.1))
    
    prior.add_parameter('C/O', dist=(0.1, 1.0))
    prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    prior.add_parameter('log_pquench', dist=(-3, 3))
    # prior.add_parameter('C_iso', dist=(10, 200))

    # prior.add_parameter('fsed', dist=(0.01, 10))
    prior.add_parameter('fsed_MgSiO3(s)_crystalline__DHS', dist=(1e-4, 10))
    prior.add_parameter('fsed_Fe(s)_crystalline__DHS', dist=(1e-4, 10))
    
    prior.add_parameter('eq_scaling_MgSiO3(s)_crystalline__DHS', dist=(-10, 1))
    prior.add_parameter('eq_scaling_Fe(s)_crystalline__DHS', dist=(-10, 1))
    
    prior.add_parameter('sigma_lnorm', dist=(1.005, 3))
    prior.add_parameter('logKzz', dist=(4, 14))

    # prior.add_parameter('corr_len', dist=(-3, 0))
    # prior.add_parameter('corr_amp', dist=(0, 1))

    # prior.add_parameter('rv', dist=(-1000, 1000))


    n_live = 480

    # run the sampler!
    # print(f'starting pool with {os.cpu_count()} cores')
    # with mp.Pool(os.cpu_count()) as pool:
    print(f'starting pool with {size} processes')
    comm.Barrier()
    with MPIPool() as pool:
        sampler = Sampler(prior, likelihood,
                          n_live=n_live,
                          filepath=checkpoint_file,
                          pool=pool,
                          n_networks=4,
                          resume=resume
                          )
        t_start = time.time()
        sampler.run(f_live=f_live, # default is 0.01, fract of evidence in live set before termination 
                    discard_exploration=discard_exploration, # true for publication ready? fully unbiased
                    verbose=True)
        t_end = time.time()
        
    if rank==0:
        print('Total time: {:.1f}s'.format(t_end - t_start))
    
    
        points, log_w, log_l = sampler.posterior()
        # log_l = log_l[~np.isnan(points)]
        # log_w = log_w[~np.isnan(points)]
        points[np.isnan(points)] = 0.0
        # print(points)
        corndog = corner.corner(
            points, weights=np.exp(log_w), 
            bins=20, labels=prior.keys, color='dodgerblue',
            plot_datapoints=False,
            range=np.repeat(0.999, len(prior.keys))
        )
        plt.savefig(output_dir+f'cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
        print('log Z: {:.2f}'.format(sampler.log_z))
    
        best = points[np.where(log_l==np.nanmax(log_l))][0]
    
        print('found best fit parameters:')
        print(best)
    
        best_params = default_params
        for i,param in enumerate(prior.keys):
            best_params[param] = best[i]
    
        test_w, test_f = spectrum_generator(best_params)
        plt.figure()
        s_test_f = spectres(w, test_w[0], test_f[0])
        plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
        plt.plot(w, s_test_f, label=ln, color='red')
        plt.legend()
        plt.savefig(output_dir+f'best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
