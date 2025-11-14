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
# rank = 0

# prt specific imports
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global, dtdp_temperature_profile
from petitRADTRANS.radtrans import Radtrans # <--- this is our "spectrum generator," e.g. the radiative transfer solver
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.clouds import return_cloud_mass_fraction, simple_cdf
from petitRADTRANS.math import filter_spectrum_with_spline


# general setup
retrieval_name = 'BENCH_FAKE_TEST'
output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
n_live = 480
discard_exploration = False
f_live = 0.05
resume = False
dyn = True

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # load data, start species if needed for photometry
    # time.sleep(rank) # sleeping here to avoid two processes trying to write to species.ini at the same time
    # SpeciesInit()

    data_path = './example_data/'
    data = np.loadtxt(data_path+'fake_data.txt')
    
    w = data[:,0]
    f = data[:,1]
    fe = data[:,2]

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron) # units of pRT models to units of data

    default_params = {
        'R_pl':1.0,
        'logg':4.0,
        'plx':10.0,

        'kappa_ir':0.01,
        'gamma':0.4,
        'T_int':850,

        'C/O':0.55,
        'Fe/H':0.0,
        'fsed':4,
        'sigma_lnorm':1.5,
        'logKzz':10,
    }

    def likelihood_nautilus(param_dict):
        w_i, f_i = spectrum_generator(param_dict)
        # downsample spectra
        rb_f_i = spectres(w, w_i, f_i)
        # compute chi2
        chi2 = np.nansum(((f - rb_f_i)/fe)**2)
        ln = -chi2/2 - np.nansum(np.log(2*np.pi*fe**2)/2) # normalize chi2 to ln
        return ln
    
    prior = Prior()
    
    mu_radius = 1.0
    sigma_radius = 0.1
    a_radius, b_radius = (0.75 - mu_radius) / sigma_radius, (2.0 - mu_radius) / sigma_radius
    prior.add_parameter('R_pl', dist=truncnorm(a_radius, b_radius, loc=mu_radius, scale=sigma_radius))
    prior.add_parameter('logg', dist=(3.0, 5.5))
    prior.add_parameter('plx', dist=norm(loc=10.0, scale=0.5))
    # prior.add_parameter('kappa_ir', dist=(1e-3, 5e-1))
    # prior.add_parameter('gamma', dist=(1e-3, 0.999))
    # prior.add_parameter('T_int', dist=(500, 1500))
    # prior.add_parameter('C/O', dist=(0.1, 1.0))
    # prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    # prior.add_parameter('fsed', dist=(0.01, 10))
    # prior.add_parameter('sigma_lnorm', dist=(1.005, 3))
    # prior.add_parameter('logKzz', dist=(4, 14))

    def likelihood_dyn(cube):
        w_i, f_i = spectrum_generator(cube)
        # downsample spectra
        rb_f_i = spectres(w, w_i, f_i)
        # compute chi2
        chi2 = np.nansum(((f - rb_f_i)/fe)**2)
        ln = -chi2/2 - np.nansum(np.log(2*np.pi*fe**2)/2) # normalize chi2 to ln
        return ln
    
    def prior_dyn(cube):
        cube = prior.unit_to_physical(cube)
        return cube


    chem = PreCalculatedEquilibriumChemistryTable()
    # Load scattering version of pRT

    rtpressures = np.logspace(-6, 3, 100) # set pressure range
    line_species = [
        'H2O',
        'CO-NatAbund',
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
        if isinstance(params, dict):
            planet_radius = params['R_pl']* cst.r_jup_mean
            reference_gravity = 1e1**params['logg']
            parallax = params['plx']
            infrared_mean_opacity = default_params['kappa_ir']
            gamma = default_params['gamma']
            intrinsic_temperature = default_params['T_int']
            co_ratio = default_params['C/O']
            feh = default_params['Fe/H']
            fsed = default_params['fsed'] # global fsed for now
            sigma_lnorm = default_params['sigma_lnorm']
            logkzz = default_params['logKzz']
            # infrared_mean_opacity = params['kappa_ir']
            # gamma = params['gamma']
            # intrinsic_temperature = params['T_int']
            # co_ratio = params['C/O']
            # feh = params['Fe/H']
            # fsed = params['fsed'] # global fsed for now
            # sigma_lnorm = params['sigma_lnorm']
            # logkzz = params['logKzz']
        else:
            planet_radius = params[0]* cst.r_jup_mean
            reference_gravity = 1e1**params[1]
            parallax = params[2]
            infrared_mean_opacity = default_params['kappa_ir']
            gamma = default_params['gamma']
            intrinsic_temperature = default_params['T_int']
            co_ratio = default_params['C/O']
            feh = default_params['Fe/H']
            fsed = default_params['fsed'] # global fsed for now
            sigma_lnorm = default_params['sigma_lnorm']
            logkzz = default_params['logKzz']
            # infrared_mean_opacity = params[3]
            # gamma = params[4]
            # intrinsic_temperature = params[5]
            # co_ratio = params[6]
            # feh = params[7]
            # sigma_lnorm = params[8]
            # logkzz = params[9]
            # fsed = params[10] # global fsed for now
        
        pressures = atmosphere.pressures * 1e-6 # cgs to bar
        r2d2 = (planet_radius/(cst.pc/(parallax/1000)))**2

        # P-T
        temperature = temperature_profile_function_guillot_global(
            pressures=pressures,
            infrared_mean_opacity=infrared_mean_opacity,
            gamma=gamma,
            gravities=reference_gravity,
            intrinsic_temperature=intrinsic_temperature,
            equilibrium_temperature=0.0 # we're doing pure emission here no star
        )

        co_ratios = co_ratio * np.ones_like(pressures)
        log10_metallicities = feh * np.ones_like(pressures)

        mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
            co_ratios=co_ratios,
            log10_metallicities=log10_metallicities,
            temperatures=temperature,
            pressures=pressures,
            full=True
        )

        # mmw = params['mmw'] # we get mean_molar_masses from chem.interpolate instead of setting it ourselves
        # mean_molar_masses = mmw * np.ones_like(temperature)
        mmw = np.mean(mean_molar_masses)

        eddy_diffusion_coefficients = np.ones_like(temperature)*1e1**logkzz
        
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

        wavelengths *= 1e4
        flux *= r2d2 * unit_conv

        return wavelengths, flux

    if rank==0:
        t_start = time.time()
        test_w, test_f = spectrum_generator(default_params)
        t_end = time.time()
        print('First spectrum Generation time: {:.1f}s'.format(t_end - t_start))
        t_start = time.time()
        ln = likelihood_nautilus(default_params)
        t_end = time.time()
        print('Likelihood time: {:.1f}s'.format(t_end - t_start))
        s_test_f = spectres(w, test_w, test_f)
        plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
        plt.plot(w, s_test_f, label=ln, color='red')
        plt.legend()
        plt.savefig(output_dir+'test_alldata_generation.png')
        
        # benchmark_model_spectrum = np.array([w,s_test_f,np.abs(np.random.normal(loc=np.nanmedian(s_test_f)/10, scale=s_test_f/100, size=len(w)))]).T
        # np.savetxt('./example_data/fake_data.txt', benchmark_model_spectrum)

        ndim = prior.dimensionality()
        test_cube = np.ones(ndim) / 2
        print(test_cube)
        test_cube = prior_dyn(test_cube)
        print(test_cube)
        ln_test = likelihood_dyn(test_cube)
        print(ln_test)


    # run the sampler!

    if dyn:

        import dynesty
        print(f'starting pool with {size} processes')
        comm.Barrier()
        with MPIPool() as pool:

            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            # "Static" nested sampling.
            sampler = dynesty.NestedSampler(likelihood_dyn, prior_dyn, ndim, pool=pool)
            t_start = time.time()
            sampler.run_nested(dlogz=f_live, 
                               checkpoint_file=checkpoint_file.replace('.hdf5','.save'),
                               resume=resume)
            t_end = time.time()
            sresults = sampler.results    

        if rank == 0:
            print('Sampling took: {:.1f}s'.format(t_end - t_start))

            from dynesty import plotting as dyplot

            lnz_truth = ndim * -np.log(2 * 10.)  # analytic evidence solution
            fig, axes = dyplot.runplot(sresults, lnz_truth=lnz_truth)  # summary (run) plot
            plt.savefig(output_dir+f'dyn_runplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            # plot extended run (res2; right)
            fg, ax = dyplot.cornerplot(sresults, color='dodgerblue', truths=np.array(default_params.items()),
                                       truth_color='black', show_titles=True,
                                       quantiles=None, max_n_ticks=3,
                                       fig=(fig, axes[:, 4:]))
            plt.savefig(output_dir+f'dyn_cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            from dynesty import utils as dyfunc

            samples, weights = sresults.samples, sresults.importance_weights()
            mean, cov = dyfunc.mean_and_cov(samples, weights)

            sresults.summary()
        
            best_params = mean
        
            test_w, test_f = spectrum_generator(best_params)
            plt.figure()
            s_test_f = spectres(w, test_w[0], test_f[0])
            plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
            plt.plot(w, s_test_f, label=ln, color='red')
            plt.legend()
            plt.savefig(output_dir+f'dyn_best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            print('found best fit parameters:')
            print(dict(zip(default_params.keys(), best_params)))



    else:
        # print(f'starting pool with {os.cpu_count()} cores')
        # with mp.Pool(os.cpu_count()) as pool:
        print(f'starting pool with {size} processes')
        comm.Barrier()
        with MPIPool() as pool:
            sampler = Sampler(prior, likelihood_nautilus,
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
            print('Sampling took: {:.1f}s'.format(t_end - t_start))
        
        
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
            plt.savefig(output_dir+f'nautlius_cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
            print('log Z: {:.2f}'.format(sampler.log_z))
        
            best = points[np.where(log_l==np.nanmax(log_l))][0]
        
            best_params = default_params
            for i,param in enumerate(prior.keys):
                best_params[param] = best[i]
        
            test_w, test_f = spectrum_generator(best_params)
            plt.figure()
            s_test_f = spectres(w, test_w[0], test_f[0])
            plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
            plt.plot(w, s_test_f, label=ln, color='red')
            plt.legend()
            plt.savefig(output_dir+f'nautilus_best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            print('found best fit parameters:')
            print(dict(zip(default_params.keys(), best)))

