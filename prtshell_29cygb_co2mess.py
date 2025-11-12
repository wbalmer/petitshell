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
from petitRADTRANS.chemistry.utils import mass_fractions2volume_mixing_ratios as mf2vmr
from petitRADTRANS.chemistry.utils import volume_mixing_ratios2mass_fractions as vmr2mf
from petitRADTRANS.chemistry.clouds import return_cloud_mass_fraction, simple_cdf
from petitRADTRANS.math import filter_spectrum_with_spline


# general setup
retrieval_name = '29cygb_shell_actuallyfixedradiusbounds'
output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
discard_exploration = False
f_live = 0.01
resume = True

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # load data, start species
    time.sleep(rank)
    SpeciesInit()

    data_path = './example_data/'
    sdata = np.loadtxt(data_path+'29cygb_charis_wm2um.dat', delimiter=',')
    gdata = fits.getdata(data_path+'29cygb_gravity.fits')
    pdata = data_path+'29cygb_photometry.dat'
    
    sw = sdata[:,0]
    sf = sdata[:,1]
    sfe = sdata[:,2]

    gw = gdata['WAVELENGTH']
    gf = gdata['FLUX']
    gcov = gdata['COVARIANCE']
    gfe = np.sqrt(np.diag(gcov))

    pnames = []
    pws = []
    pfs = []
    pfes = []
    psyns = []

    with open(pdata) as photometry:
        for line in photometry:
            # # must be the comment character
            if line[0] == '#':
                continue

            vals = line.split(',')
            pnames.append(vals[0])
            wlow = float(vals[1])
            whigh = float(vals[2])
            pws.append([0.95 * wlow, 1.05 * whigh])
            pfs.append(float(vals[3]))
            pfes.append(float(vals[4]))
            psyns.append(SyntheticPhotometry(vals[0]))

    # set up the filter spacing, e.g. the "frequencies" that we will high pass filter over
    # x_nodes = np.linspace(w[0], w[-1], 60)

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron)

    default_params = {
        'R_pl':1.3,
        'plx':37.2539,
        'logg':3.7,

        # 'kappa_ir':0.01,
        # 'gamma':0.4,
        # 'T_int':850,
        # 'T_eq':0.0,

        'T_bottom':9000.,
        'N_layers':6,
        'dPT_6':0.07,
        'dPT_5':0.10,
        'dPT_4':0.18,
        'dPT_3':0.27,
        'dPT_2':0.24,
        'dPT_1':0.25,

        'C/O':0.55,
        'Fe/H':0.75,
        'log_pquench':1.0,
        # 'C_iso':100,
        'fsed':4,
        'sigma_lnorm':1.5,
        'logKzz':10,
        # 'mmw':2.33,
        'corr_len_sphere':-1, # log10 [-3, 0] 
        'corr_amp_sphere':0.5 # [0, 1]
    }

    def likelihood(param_dict, debug=False):

        ln = 0

        params = default_params
        for param in param_dict:
            params[param] = param_dict[param]

        w_i, f_i = spectrum_generator(params)

        # compute sphere likelihood:

        rb_f_i = spectres(sw, w_i[0], f_i[0])
        
        if debug:
            plt.figure()
            plt.plot(sw, rb_f_i)

        if 'corr_len_sphere' in param_dict.keys():
            # from Wang et al. 2020, species.fit.fit_model
            wavel_j, wavel_i = np.meshgrid(sw, sw)

            error_j, error_i = np.meshgrid(sfe, sfe)

            corr_len = 10.0 ** param_dict["corr_len_sphere"]  # (um)
            corr_amp = param_dict["corr_amp_sphere"]

            cov_matrix = (
                corr_amp**2
                * error_i
                * error_j
                * np.exp(-((wavel_i - wavel_j) ** 2) / (2.0 * corr_len**2))
                + (1.0 - corr_amp**2) * np.eye(sw.shape[0]) * error_i**2
            )

            ln_s = (
                (sf - rb_f_i)
                @ np.linalg.inv(cov_matrix)
                @ (sf - rb_f_i)
            )

            ln_s += np.nansum(
                np.log(2.0 * np.pi * sfe**2)
            )

            ln_s *= -0.5

            ln += ln_s
        
        else:
            chi2 = np.nansum(((sf - rb_f_i)/sfe)**2)
            ln += -chi2/2 - np.nansum(np.log(2*np.pi*sfe**2)/2)

        # compute gravity likelihood

        rb_f_i = spectres(gw, w_i[1], f_i[1])
        
        if debug:
            plt.plot(gw, rb_f_i)

        ln_g = (
                (gf - rb_f_i)
                @ np.linalg.inv(gcov)
                @ (gf - rb_f_i)
            )

        ln_g += np.nansum(
            np.log(2.0 * np.pi * gfe**2)
        )

        ln_g *= -0.5

        ln += ln_g

        # compute photometry likelihood
        
        ln_p = 0
        
        for i in range(len(pnames)):

            chi2 = np.nansum(((pfs[i] - f_i[2][i])/pfes[i])**2)
            ln_p_i = -chi2/2 - np.nansum(np.log(2*np.pi*pfes[i]**2)/2)

            ln_p += ln_p_i
            
        ln += ln_p
            
        if debug:
            plt.errorbar(w_i[2], f_i[2], marker='s', color='red')
            plt.savefig('temp_spec.png')
            print(ln_s, ln_g, ln_p)

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

    smresl = '1000' # sphere model resolution, R=160 c-k
    atmosphere_sphere = Radtrans(
        pressures = rtpressures,
        line_species = [i+f'.R{smresl}' for i in line_species],
        rayleigh_species = rayleigh_species, # why is the sky blue?
        gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
        cloud_species = cloud_species, # these will be important for clouds
        wavelength_boundaries = [sw[0]-0.1, sw[-1]+0.1],
        line_opacity_mode='c-k' # lbl or c-k
    )

    gmresl = '1000' # gravity model resolution, standard R=1000 c-k
    atmosphere_gravity = Radtrans(
        pressures = rtpressures,
        line_species = [i+f'.R{gmresl}' for i in line_species],
        rayleigh_species = rayleigh_species, # why is the sky blue?
        gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
        cloud_species = cloud_species, # these will be important for clouds
        wavelength_boundaries = [gw[0]-0.1, gw[-1]+0.1],
        line_opacity_mode='c-k' # lbl or c-k
    )

    atmosphere_photometrys = []
    ptmresl = '1000' # photometry model resolution, R=40 c-k
    for i,filt in enumerate(pnames):
        atmosphere_phot_i = Radtrans(
            pressures = rtpressures,
            line_species = [i+f'.R{ptmresl}' for i in line_species],
            rayleigh_species = rayleigh_species, # why is the sky blue?
            gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
            cloud_species = cloud_species, # these will be important for clouds
            wavelength_boundaries = pws[i],
            line_opacity_mode='c-k' # lbl or c-k
        )
        atmosphere_photometrys.append(atmosphere_phot_i)


    def spectrum_generator(params, quench_co2_off_co=False, debug_abund=False):
        planet_radius = params['R_pl']* cst.r_jup_mean
        parallax = params['plx']
        r2d2 = (planet_radius/(cst.pc/(parallax/1000)))**2
        if 'mass' in params.keys():
            reference_gravity = (cst.G*params['mass']*cst.m_jup)/(planet_radius**2)
            # print('ref grav is', reference_gravity, 'which is logg=', np.log10(reference_gravity))
        else:
            reference_gravity = 1e1**params['logg']
        
        pressures = atmosphere_sphere.pressures * 1e-6 # cgs to bar

        # gradient
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
            top_of_atmosphere_pressure=-3,
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

        if debug_abund:
            mf_orig = copy.deepcopy(mass_fractions)

        if quench_co2_off_co:
            # vmrs = mf2vmr(mass_fractions, mean_molar_masses)
            quench_idx = pressures <= 10**log_pquench
            # vmr_h2 = vmrs['H2'][quench_idx]
            # vmr_co = vmrs['CO'][quench_idx]
            # vmr_h2o = vmrs['H2O'][quench_idx]
            mf_h2 = mass_fractions['H2'][quench_idx]
            mf_co = mass_fractions['CO'][quench_idx]
            mf_h2o = mass_fractions['H2O'][quench_idx]
            Keq = 18.3*np.exp((-2376/temperature[quench_idx]) - ((932/temperature[quench_idx])**2))
            # vmrs['CO2'][quench_idx] = Keq * (vmr_co * vmr_h2o)/(vmr_h2)
            # mass_fractions = mf2vmr(vmrs, mean_molar_masses)
            mass_fractions['CO2'][quench_idx] = (mf_co * mf_h2o)/(mf_h2* Keq)

        if debug_abund:
            mf_list = ['H2', 'H2O', 'CO', 'CH4', 'CO2']
            mf_colors = ['blue', 'red', 'green', 'orange', 'purple']
            plt.figure()
            i = 0
            for key in list(mass_fractions.keys()):
                if key in mf_list:
                    if key == 'CO2':
                        plt.plot(mass_fractions[key], pressures, label=key, color='k', ls='--')
                        plt.plot(mf_orig[key], pressures, color='k')
                    else:
                        i += 1
                        plt.plot(mass_fractions[key], pressures, label=key, ls='--', color=mf_colors[i])
                        plt.plot(mf_orig[key], pressures, color=mf_colors[i])
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(1e3, 1e-6)
            plt.savefig(output_dir+'debug_abund.png')

        if '13CO' in atmosphere_sphere.line_species:
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
            for specie in atmosphere_sphere.cloud_species:
                cloud_f_sed[specie] = params[f'fsed_{specie}']
        else:
            fsed = params['fsed'] # global fsed for now
            cloud_f_sed = {specie:fsed for specie in atmosphere_sphere.cloud_species}
        cloud_particle_radius_distribution_std = sigma_lnorm

        cbases = {}
        for specie in atmosphere_sphere.cloud_species:
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
        smfs = copy.copy(mass_fractions)
        gmfs = copy.copy(mass_fractions)
        ptmfs = copy.copy(mass_fractions) 
        for key in line_species:
            smfs[key+f'.R{smresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            smfs.pop(key.split('_')[0].split('-')[0].split(".")[0])
            gmfs[key+f'.R{gmresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            gmfs.pop(key.split('_')[0].split('-')[0].split(".")[0], None)
            ptmfs[key+f'.R{ptmresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            ptmfs.pop(key.split('_')[0].split('-')[0].split(".")[0], None)

        # set up resolution and wavelength range specific R-T calcs for each spectrum/dataset
        wavelengths_sphere, flux_sphere, additional_returned_quantities = atmosphere_sphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=smfs,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_f_sed=cloud_f_sed,
            cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
            return_contribution=False
        )

        wavelengths_gravity, flux_gravity, additional_returned_quantities = atmosphere_gravity.calculate_flux(
            temperatures=temperature,
            mass_fractions=gmfs,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_f_sed=cloud_f_sed,
            cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
            return_contribution=False
        )

        phot_wavels = []
        phot_fluxes = []
        for i in range(len(pnames)):
            wavelengths_phot_i, flux_gravity_phot_i, additional_returned_quantities = atmosphere_photometrys[i].calculate_flux(
                temperatures=temperature,
                mass_fractions=ptmfs,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                eddy_diffusion_coefficients=eddy_diffusion_coefficients,
                cloud_f_sed=cloud_f_sed,
                cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
                return_contribution=False
            )
            synphot = psyns[i]

            integrated_flux, _ = synphot.spectrum_to_flux(wavelengths_phot_i*1e4, flux_gravity_phot_i* r2d2 * unit_conv) # careful of output here, bc it gives tuple flux, err

            phot_wavels.append(np.nanmean(wavelengths_phot_i*1e4))
            phot_fluxes.append(integrated_flux)

        wavelengths = [wavelengths_sphere*1e4, wavelengths_gravity*1e4, phot_wavels]
        flux = [flux_sphere * r2d2 * unit_conv, flux_gravity * r2d2 * unit_conv, phot_fluxes]

        # wavelengths = np.concatenate((wavelengths_sphere, wavelengths_gravity), axis=0)
        # flux = np.concatenate((flux_sphere, flux_gravity), axis=0)

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
        s_test_f = spectres(sw, test_w[0], test_f[0])
        plt.errorbar(sw, sf, yerr=sfe, label='sphere', marker='.', color='k', ls='none')
        g_test_f = spectres(gw, test_w[1], test_f[1])
        plt.errorbar(gw, gf, yerr=gfe, label='gravity', color='k', ls='none')
        plt.plot(sw, s_test_f, label=ln, color='red')
        plt.plot(gw, g_test_f, color='red')
        plt.errorbar(test_w[2], pfs, yerr=pfes, marker='o', color='k', label='photometry')
        plt.errorbar(test_w[2], test_f[2], marker='s', color='red')
        plt.legend()
        plt.savefig(output_dir+'test_alldata_generation.png')

        test_w, test_f_qco2 = spectrum_generator(default_params, quench_co2_off_co=True, debug_abund=True)
        plt.figure()
        plt.plot(test_w[2], test_f[2])
        plt.plot(test_w[2], test_f_qco2[2])
        plt.xlim(3.5, 5.5)
        plt.savefig(output_dir+'test_co2_quench.png')

    prior = Prior()
    
    mu_radius = 1.3
    sigma_radius = 0.1
    a_radius, b_radius = (0.75 - mu_radius) / sigma_radius, (2.0 - mu_radius) / sigma_radius
    prior.add_parameter('R_pl', dist=truncnorm(a_radius, b_radius, loc=mu_radius, scale=sigma_radius))
    
    mu_mass = 15
    sigma_mass = 5
    a_mass, b_mass = (0.1 - mu_mass) / sigma_mass, (50.0 - mu_mass) / sigma_mass
    prior.add_parameter('mass', dist=truncnorm(a_mass, b_mass, mu_mass, scale=sigma_mass))
    
    # prior.add_parameter('logg', dist=norm(loc=3.7, scale=0.1))
    
    prior.add_parameter('plx', dist=norm(loc=24.5456, scale=0.0911))

    prior.add_parameter('T_bottom', dist=(2500, 25000))
    prior.add_parameter('dPT_1', dist=norm(loc=0.25, scale=0.025))
    prior.add_parameter('dPT_2', dist=norm(loc=0.25, scale=0.045))
    prior.add_parameter('dPT_3', dist=norm(loc=0.26, scale=0.05))
    prior.add_parameter('dPT_4', dist=norm(loc=0.2, scale=0.05))
    prior.add_parameter('dPT_5', dist=norm(loc=0.12, scale=0.045))
    prior.add_parameter('dPT_6', dist=norm(loc=0.07, scale=0.07))
    
    prior.add_parameter('C/O', dist=(0.1, 1.0))
    prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    # prior.add_parameter('C/O', dist=norm(loc=0.55, scale=0.05))
    # prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    prior.add_parameter('log_pquench', dist=(-3, 3))
    # prior.add_parameter('C_iso', dist=(10, 200))

    # prior.add_parameter('fsed', dist=(0.01, 10))
    prior.add_parameter('fsed_MgSiO3(s)_crystalline__DHS', dist=(1e-4, 10))
    prior.add_parameter('fsed_Fe(s)_crystalline__DHS', dist=(1e-4, 10))
    
    prior.add_parameter('eq_scaling_MgSiO3(s)_crystalline__DHS', dist=(-10, 1))
    prior.add_parameter('eq_scaling_Fe(s)_crystalline__DHS', dist=(-10, 1))
    
    prior.add_parameter('sigma_lnorm', dist=(1.005, 3))
    prior.add_parameter('logKzz', dist=(4, 14))

    prior.add_parameter('corr_len_sphere', dist=(-3, 0))
    prior.add_parameter('corr_amp_sphere', dist=(0, 1))

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
        s_test_f = spectres(sw, test_w[0], test_f[0])
        plt.errorbar(sw, sf, yerr=sfe, label='sphere', marker='.', color='k', ls='none')
        g_test_f = spectres(gw, test_w[1], test_f[1])
        plt.errorbar(gw, gf, yerr=gfe, label='gravity', color='k', ls='none')
        plt.plot(sw, s_test_f, label=ln, color='red')
        plt.plot(gw, g_test_f, color='red')
        plt.errorbar(test_w[2], pfs, yerr=pfes, marker='o', color='k', label='photometry')
        plt.errorbar(test_w[2], test_f[2], marker='s', color='red')
        plt.legend()
        plt.savefig(output_dir+f'best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
