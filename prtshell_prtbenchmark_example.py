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

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
rank = 0

# prt specific imports
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_ret_model, dtdp_temperature_profile
from petitRADTRANS.radtrans import Radtrans # <--- this is our "spectrum generator," e.g. the radiative transfer solver
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.clouds import return_cloud_mass_fraction, simple_cdf
from petitRADTRANS.math import filter_spectrum_with_spline


# general setup
retrieval_name = 'BENCH_DYNESTY_FIXLARGEPARAM'
output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
n_live = 1000
discard_exploration = False
f_live = 0.01
networks = 4

resume = False

dyn = True

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # load data, start species if needed for photometry
    time.sleep(rank) # sleeping here to avoid two processes trying to read/write to init files at the same time
    # SpeciesInit()

    data_path = './example_data/'
    # data = np.loadtxt(data_path+'fake_data.txt')
    data = np.loadtxt(data_path+'fake_data_1-3mu.txt')

    w = data[:,0]
    f = data[:,1]
    fe = data[:,2]

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron) # units of pRT models to units of data

    default_params = {
        'R_pl':1.0,
        'logg':4.0,
        'plx':10.0,

        'T_int':1200,
        'T1':0.5,
        'T2':0.5,
        'T3':0.5,
        'log_delta':0.7,
        'alpha':1.5,

        'C/O':0.55,
        'Fe/H':0.0,
        'log_pquench':1.0,
        'fsed':1,
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

    prior.add_parameter('T3', dist=(0, 1))
    prior.add_parameter('T2', dist=(0, 1))
    prior.add_parameter('T1', dist=(0, 1))
    prior.add_parameter('T_int', dist=(200, 2000))
    prior.add_parameter('log_delta', dist=(0, 1))
    prior.add_parameter('alpha', dist=(1, 2))

    prior.add_parameter('C/O', dist=(0.1, 1.0))
    prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    prior.add_parameter('log_pquench', dist=(-3,3))
    prior.add_parameter('fsed', dist=(0.01, 10))
    prior.add_parameter('sigma_lnorm', dist=(1.005, 3))
    prior.add_parameter('logKzz', dist=(4, 14))

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
    chem.load()
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

            t3 = params['T3']
            t2 = params['T2']
            t1 = params['T1']
            intrinsic_temperature = params['T_int']
            log_delta = params['log_delta']
            alpha = params['alpha']

            co_ratio = params['C/O']
            feh = params['Fe/H']
            log_pquench = params['log_pquench']
            fsed = params['fsed'] # global fsed for now
            sigma_lnorm = params['sigma_lnorm']
            logkzz = params['logKzz']
        else:
            planet_radius = params[0]* cst.r_jup_mean
            reference_gravity = 1e1**params[1]
            parallax = params[2]

            t3 = params[3]
            t2 = params[4]
            t1 = params[5]
            intrinsic_temperature = params[6]
            log_delta = params[7]
            alpha = params[8]

            co_ratio = params[9]
            feh = params[10]
            log_pquench = params[11]
            fsed = params[12] # global fsed for now
            sigma_lnorm = params[13]
            logkzz = params[14]

        pressures = atmosphere.pressures * 1e-6 # cgs to bar
        r2d2 = (planet_radius/(cst.pc/(parallax/1000)))**2

        # P-T
        # Priors for these parameters are implemented here, as they depend on each other
        t3 = ((3. / 4. * intrinsic_temperature ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - t3)
        t2 = t3 * (1.0 - t2)
        t1 = t2 * (1.0 - t1)
        delta = ((10.0 ** (-3.0 + 5.0 * log_delta)) * 1e6) ** (-alpha)

        rad_trans_params = [
            np.array([t1,t2,t3]),
            delta,
            alpha,
            intrinsic_temperature,
            pressures,
            True,
            co_ratio,
            feh
        ]
        temperature = temperature_profile_function_ret_model(rad_trans_params)

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
        plt.errorbar(w, f, yerr=np.abs(fe), label='jwst', marker='.', color='k', ls='none')
        plt.plot(w, s_test_f, label=ln, color='red')
        plt.legend()
        plt.savefig(output_dir+'test_alldata_generation.png')

        # generate fake spectra
        # snr = 10
        # noise_sim = np.abs(s_test_f/snr)

        # benchmark_model_spectrum = np.array([w,s_test_f,noise_sim]).T
        # np.savetxt('./example_data/fake_data_1-3mu.txt', benchmark_model_spectrum)

        # plt.figure()
        # plt.errorbar(benchmark_model_spectrum[:,0], benchmark_model_spectrum[:,1], yerr=benchmark_model_spectrum[:,2], label=f'fake data snr={snr}', marker='.', color='k', ls='none')
        # plt.legend()
        # plt.savefig(output_dir+'test_fakedata_generation.png')

    # run the sampler!

    if dyn:

        import dynesty

        ndim = prior.dimensionality()
        test_cube = np.ones(ndim) / 2
        print(test_cube)
        test_cube = prior_dyn(test_cube)
        print(test_cube)
        ln_test = likelihood_dyn(test_cube)
        print(ln_test)

        print(f'starting pool with {os.cpu_count()} cores')
        with mp.Pool(os.cpu_count()) as pool:
        # print(f'starting pool with {size} processes')
        # comm.Barrier()
        # with MPIPool() as pool:

            # if not pool.is_master():
            #     pool.wait()
            #     sys.exit(0)
            # "Static" nested sampling.
            if resume:
                sampler = dynesty.DynamicNestedSampler.restore(
                    fname=checkpoint_file.replace('.hdf5','.save'),
                    pool=pool,
                )
            else:
                sampler = dynesty.DynamicNestedSampler(likelihood_dyn,
                                                       prior_dyn,
                                                       ndim,
                                                       sample='unif',
                                                       pool=pool, 
                                                       queue_size=os.cpu_count())
            t_start = time.time()
            sampler.run_nested(
                               n_effective=10000,
                               nlive_init=n_live,
                               dlogz_init=f_live,
                               checkpoint_file=checkpoint_file.replace('.hdf5','.save'),
                               resume=resume)
            t_end = time.time()
            sresults = sampler.results

        if rank == 0:
            print('Sampling took: {:.1f}s'.format(t_end - t_start))

            from dynesty import plotting as dyplot

            # fig, axes = dyplot.runplot(sresults)  # summary (run) plot
            # plt.savefig(output_dir+f'dyn_runplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            print(np.array(list(default_params.items())))

            # plot extended run (res2; right)
            fg, ax = dyplot.cornerplot(sresults, color='dodgerblue', truths=list(default_params.items()),
                                       truth_color='black', show_titles=True, labels=list(default_params.keys()),
                                       quantiles=None, max_n_ticks=3
                                       )
            plt.savefig(output_dir+f'dyn_cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            from dynesty import utils as dyfunc

            samples, weights = sresults.samples, sresults.importance_weights()
            median = np.median(samples, axis=0)
            quant = np.quantile(samples, [0.1587, 0.8413], axis=0)

            sresults.summary()

            best_params = median
            print(best_params)

            test_w, test_f = spectrum_generator(best_params)
            print('test w is', test_w)
            print('test f is', test_f)
            plt.figure()
            s_test_f = spectres(w, test_w, test_f)
            plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
            plt.plot(test_w, test_f, label=ln, color='red')
            plt.legend()
            plt.savefig(output_dir+f'dyn_best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            print('found best fit parameters:')
            print(dict(zip(default_params.keys(), best_params)))



    else:
        print(f'starting pool with {os.cpu_count()} cores')
        with mp.Pool(os.cpu_count()) as pool:
        # print(f'starting pool with {size} processes')
        # comm.Barrier()
        # with MPIPool() as pool:
            sampler = Sampler(prior, likelihood_nautilus,
                            n_live=n_live,
                            filepath=checkpoint_file,
                            pool=pool,
                            n_networks=networks,
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
            s_test_f = spectres(w, test_w, test_f)
            plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
            plt.plot(w, s_test_f, label=ln, color='red')
            plt.legend()
            plt.savefig(output_dir+f'nautilus_best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

            print('found best fit parameters:')
            print(dict(zip(default_params.keys(), best)))
