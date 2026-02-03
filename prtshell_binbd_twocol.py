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
from scipy.stats import norm, truncnorm, loguniform
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
from petitRADTRANS.physics import temperature_profile_function_guillot_global, dtdp_temperature_profile, temperature_profile_function_ret_model, frequency2wavelength
from petitRADTRANS.radtrans import Radtrans # <--- this is our "spectrum generator," e.g. the radiative transfer solver
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import mass_fractions2volume_mixing_ratios as mf2vmr
from petitRADTRANS.chemistry.utils import volume_mixing_ratios2mass_fractions as vmr2mf
from petitRADTRANS.chemistry.clouds import return_cloud_mass_fraction, simple_cdf
from petitRADTRANS.math import filter_spectrum_with_spline
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
from petitRADTRANS.fortran_convolve import fortran_convolve as fconvolve


# general setup

pmn = True
mpied = True

if pmn:
    retrieval_name = 'hd47127b_shell_twosurface_full_rvsplit_pmn'
else:
    retrieval_name = 'hd47127b_shell_twosurface_nautilus'

output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
discard_exploration = False
f_live = 0.01
n_live = 1000
resume = False
plot = False

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # load data, start species
    time.sleep(0.5*rank)
    SpeciesInit()

    data_path = './example_data/'
    gdata = fits.getdata(data_path+'test_hd47127b_nirspec.fits')
    nirspec_resolution = np.loadtxt(data_path+'hd47127B_resolution.txt')
    pdata = data_path+'hd47127b_photometry.txt'

    nsw = gdata['WAVELENGTH']
    print('wl shape '+str(len(nsw)))
    nsf = gdata['FLUX']
    nscov = gdata['COVARIANCE']
    nsfe = np.sqrt(np.diag(nscov))

    nsw_bins = np.zeros_like(nsw)
    nsw_bins[:-1] = np.diff(nsw)
    nsw_bins[-1] = nsw[-2]

    x_nodes = np.linspace(nsw[0], nsw[-1], 60)

    plt.figure()
    plt.plot(nsw, nirspec_resolution)
    plt.savefig(output_dir+'resolution.png')

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

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron)

    default_params = {

        'M_tot':105,
        'M_ratio':0.45,
        'plx':37.5,
        'C/O':0.55,
        'Fe/H':0.0,
        'C_iso':100,
        'e_hat': 2.0,

        # A
        'R_pl_A':1.0,
        # 'logg_A':5.0,

        'T_int_A':1200,
        # 'T1_A':0.5,
        # 'T2_A':0.5,
        # 'T3_A':0.5,
        'log_kappa_IR_A':0.7,
        'gamma_A':1.5,

        'log_kzz_chem_A':10,
        'fsed_A':1,
        'sigma_lnorm_A':1.5,
        'log_kzz_cloud_A':10,

        'rv_A':-10,

        # B
        'R_pl_B':1.0,
        # 'logg_B':5.0,

        'T_int_B':1200,
        # 'T1_B':0.5,
        # 'T2_B':0.5,
        # 'T3_B':0.5,
        'log_kappa_IR_B':0.7,
        'gamma_B':1.5,

        'log_kzz_chem_B':10,
        'fsed_B':1,
        'sigma_lnorm_B':1.5,
        'log_kzz_cloud_B':10,
        
        'rv_B':+10,
        
    }

    def likelihood(param_dict, debug=False):

        ln = 0

        # params = default_params
        # for param in param_dict:
        #     params[param] = param_dict[param]
        global_params = ['plx', 'C/O', 'Fe/H', 'C_iso', 'e_hat']
        
        # this is hacky, but just rejection sampling for sign of RV:
        # that is, let "A" always be the body moving blueshifted towards us
        if param_dict["rv_A"] > param_dict["rv_B"]:
            return -np.inf

        # if debug:
        #     print(param_dict)

        params_bd1 = {}
        params_bd2 = {}
        for key in param_dict.keys():
            if key in global_params:
                params_bd1[key] = param_dict[key]
                params_bd2[key] = param_dict[key]
            else:
                if "_A" in key:
                    key_a = key.split("_A")[0]
                    params_bd1[key_a] = param_dict[key]
                elif "_B" in key:
                    key_b = key.split("_B")[0]
                    params_bd2[key_b] = param_dict[key]
        if 'logg' not in params_bd1.keys():
            if 'M_tot' and 'M_ratio' in param_dict.keys():
                # a over b, so A = tot-B
                M_b = param_dict['M_tot']*param_dict['M_ratio']
                M_a = param_dict['M_tot']-M_b
                params_bd1['mass'] = M_a
                params_bd2['mass'] = M_b
        
        # if debug:
        #     print("params bd1 ")
        #     print(params_bd1)
        #     print("params bd2 ")
        #     print(params_bd2)
        
        w_i, f_i_1 = spectrum_generator(params_bd1)
        _, f_i_2 = spectrum_generator(params_bd2)

        f_i = f_i_1 + f_i_2

        if debug:
            plt.figure(figsize=(9,3))
            # plt.plot(nsw, nsf, color='k')

        # compute nirspec likelihood

        # resample to wl grid
        if debug:
            plt.plot(w_i[0], f_i[0])
        # rb_f_i = spectres(nsw, w_i[0], cv_f_i) # TODO: use frebin instead of spectres
        rb_f_i = frebin.rebin_spectrum(w_i[0], f_i[0], nsw) # input w, input f, output w
        # print('rebinned spectrum')
        if debug:
            plt.plot(nsw, rb_f_i)

        # nirspec_resolution_array = np.interp(w_i[0], nsw, nirspec_resolution)
        # convolve to resolution # TODO: use fortran convolve
        # cv_f_i = convolve(w_i[0], f_i[0], nirspec_resolution_array)
        cv_f_i = fconvolve.variable_width_convolution(nsw, rb_f_i, nirspec_resolution) # input w, input f, res array
        # print('convolved spectrum')
        if debug:
            plt.plot(nsw, cv_f_i)
        
        # subtract continuum
        frb_f_i = filter_spectrum_with_spline(nsw,cv_f_i,x_nodes=x_nodes)
        # print('continuum subtracted spectrum')

        if np.isnan(np.sum(frb_f_i)):
            return -np.inf
        
        if debug:
            plt.plot(nsw, frb_f_i, color='red', ls='-')

        ln_ns = (
                (nsf - frb_f_i)
                @ np.linalg.inv(nscov)
                @ (nsf - frb_f_i)
            )
        # print('did the inverse')
        
        if 'e_hat' in params_bd1:
            e_hat = params_bd1['e_hat']
        else:
            e_hat = 1

        ln_ns += np.nansum(
            np.log(2.0 * np.pi * (nsfe*e_hat)**2)
        )

        ln_ns *= -0.5

        ln += ln_ns

        # compute photometry likelihood
        
        ln_p = 0
        
        for i in range(len(pnames)):

            chi2 = np.nansum(((pfs[i] - f_i[1][i])/pfes[i])**2)
            ln_p_i = -chi2/2 - np.nansum(np.log(2*np.pi*pfes[i]**2)/2)

            ln_p += ln_p_i
        # print('summed the photometry')
            
        ln += ln_p
            
        if debug:
            # plt.errorbar(w_i[1], f_i[1], marker='s', color='red')
            # plt.errorbar(w_i[1], pfs, yerr=pfes, marker='o', color='k')
            # plt.xlim(4.4, 4.8)
            plt.savefig(output_dir+'temp_spec.png')
            # print(ln_ns, ln_p)
        
        # print('done with likelihood calc')
        
        if np.isnan(ln):
            return -np.inf
        else:
            return ln


    chem = PreCalculatedEquilibriumChemistryTable()
    chem.load()
    # Load scattering version of pRT

    rtpressures = np.logspace(-6, 3, 100) # set pressure range
    line_species = [
        'H2O__POKAZATEL',
        # 'CO-NatAbund',
        '12CO',
        '13CO',
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
    cloud_species = [
                     'Na2S(s)_crystalline__DHS',
                     'MgSiO3(s)_crystalline__DHS',
                     'Fe(s)_crystalline__DHS'
                    ] # these will be important for clouds

    nsmresl = '1e6' # lbl here
    nirspec_model_resl = 10000
    print('taking every '+str(int(float(nsmresl)/nirspec_model_resl))+"th lbl opacity")
    atmosphere_nirspec = Radtrans(
        pressures = rtpressures,
        line_species = [i+f'.R{nsmresl}' for i in line_species],
        rayleigh_species = rayleigh_species, # why is the sky blue?
        gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
        cloud_species = cloud_species, # these will be important for clouds
        wavelength_boundaries = [nsw[0]-0.05, nsw[-1]+0.05],
        line_opacity_mode='lbl', # lbl or c-k
        line_by_line_opacity_sampling = int(float(nsmresl)/nirspec_model_resl),
    )

    # w_lbl = frequency2wavelength(atmosphere_nirspec._frequencies)*1e4
    # nirspec_resolution_array = np.interp(w_lbl, nsw, nirspec_resolution) # native resolution

    atmosphere_photometrys = []
    ptmresl = '40' # photometry model resolution, R=40 c-k
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
    
    if plot:
        atmosphere_plot = atmosphere_nirspec

    def kzz_to_co_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem, log10_metallicities):
        # Pressure scale height (m)
        h_scale = cst.kB * temperature / (mean_molar_masses * cst.amu * reference_gravity)

        # Diffusion coefficient (m2 s-1)
        chem_kzz = 10.0**log_kzz_chem

        # Mixing timescale (s)
        t_mix = h_scale**2 / chem_kzz

        # chemical timescales eq. 12-14 from Zahnle & Marley 2014
        metal = 10.0**log10_metallicities
        # t_chem_co = 1.5e-6 * pressures**-1.0 * metal**-0.7 * np.exp(42000.0 / temperature)
        t_chem_1 = 1.5e-6 * pressures**-1.0 * metal**-0.7 * np.exp(42000.0 / temperature)
        t_chem_2 = 40 * pressures**-2.0 * np.exp(25000.0 / temperature)

        t_chem_co = ((1/t_chem_1)+(1/t_chem_2))**-1.0

        # Determine pressure at which t_mix = t_chem

        t_diff = t_mix - t_chem_co
        diff_product = t_diff[1:] * t_diff[:-1]

        # If t_mix and t_chem intersect then there
        # is 1 negative value in diff_product
        indices = diff_product < 0.0

        if np.sum(indices) == 1:
            p_quench = (pressures[1:] + pressures[:-1])[indices] / 2.0
            p_quench = p_quench[0]

        elif np.sum(indices) == 0:
            p_quench = None
        
        else:
            # print('found multiple p_quench intersections')
            # print(dict(zip(pressures, indices)))
            crossing = np.where(indices)[0]
            # print(crossing)
            p_quench = (pressures[1:] + pressures[:-1])[crossing] / 2.0
            # print(p_quench)
            p_quench = np.max(p_quench)
            # print(p_quench)
            # crash
        return p_quench
    
    def kzz_to_co2_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem, log10_metallicities):
        # Pressure scale height (m)
        h_scale = cst.kB * temperature / (mean_molar_masses * cst.amu * reference_gravity)

        # Diffusion coefficient (m2 s-1)
        chem_kzz = 10.0**log_kzz_chem

        # Mixing timescale (s)
        t_mix = h_scale**2 / chem_kzz

        # chemical timescales eq. 12-14 from Zahnle & Marley 2014
        metal = 10.0**log10_metallicities
        t_chem_co2 = 1e-10 * pressures**-0.5 * np.exp(38000.0 / temperature)

        # Determine pressure at which t_mix = t_chem

        t_diff = t_mix - t_chem_co2
        diff_product = t_diff[1:] * t_diff[:-1]

        # If t_mix and t_chem intersect then there
        # is 1 negative value in diff_product
        indices = diff_product < 0.0

        if np.sum(indices) == 1:
            p_quench = (pressures[1:] + pressures[:-1])[indices] / 2.0
            p_quench = p_quench[0]

        elif np.sum(indices) == 0:
            p_quench = None
        
        else:
            crossing = np.where(indices)[0]
            p_quench = (pressures[1:] + pressures[:-1])[crossing] / 2.0
            p_quench = np.max(p_quench)

        return p_quench

    def spectrum_generator(params, quench_co2_off_co=True, return_extras=False):
        planet_radius = params['R_pl'] * cst.r_jup_mean
        parallax = params['plx']
        r2d2 = (planet_radius/(cst.pc/(parallax/1000)))**2
        if 'mass' in params.keys():
            reference_gravity = (cst.G*params['mass']*cst.m_jup)/(planet_radius**2)
        else:
            reference_gravity = 1e1**params['logg']
        
        pressures = atmosphere_nirspec.pressures * 1e-6 # cgs to bar
        co_ratio = params['C/O']
        feh = params['Fe/H']

        # gradient
        # t_bottom = params['T_bottom']
        # num_layer = params['N_layers']
        # layer_pt_slopes = np.ones(num_layer) * np.nan
        # for index in range(num_layer):
        #     layer_pt_slopes[index] = params[f'dPT_{num_layer - index}']
        # if num_layer > 6:
        #     top_press = -6
        # else:
        #     top_press = -3
        # temperature = dtdp_temperature_profile(
        #     pressures,
        #     num_layer,
        #     layer_pt_slopes,
        #     t_bottom,
        #     top_of_atmosphere_pressure=top_press,
        #     bottom_of_atmosphere_pressure=3
        # )

        # molliere 
        # t3 = params['T3']
        # t2 = params['T2']
        # t1 = params['T1']
        # intrinsic_temperature = params['T_int']
        # log_delta = params['log_delta']
        # alpha = params['alpha']
        # # Priors for these parameters are implemented here, as they depend on each other
        # t3 = ((3. / 4. * intrinsic_temperature ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - t3)
        # t2 = t3 * (1.0 - t2)
        # t1 = t2 * (1.0 - t1)
        # delta = ((10.0 ** (-3.0 + 5.0 * log_delta)) * 1e6) ** (-alpha)

        # rad_trans_params = [
        #     np.array([t1,t2,t3]),
        #     delta,
        #     alpha,
        #     intrinsic_temperature,
        #     pressures,
        #     True,
        #     co_ratio,
        #     feh
        # ]
        # temperature = temperature_profile_function_ret_model(rad_trans_params)

        # guillot 
        T_int = params['T_int']
        T_eq = 3.0
        log_kap_ir = params['log_kappa_IR']
        gamma = params['gamma']
        temperature = temperature_profile_function_guillot_global(
            pressures,
            10 ** log_kap_ir,
            gamma,
            reference_gravity,
            T_int,
            T_eq
        )

        log_kzz_chem = params['log_kzz_chem']

        co_ratios = co_ratio * np.ones_like(pressures)
        log10_metallicities = feh * np.ones_like(pressures)

        mmw_init = np.ones_like(pressures) * 2.33

        p_quench = kzz_to_co_pquench(temperature, pressures, mmw_init, reference_gravity, log_kzz_chem, log10_metallicities)

        mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
            co_ratios=co_ratios,
            log10_metallicities=log10_metallicities,
            temperatures=temperature,
            pressures=pressures,
            carbon_pressure_quench=p_quench,
            full=True
        )

        mmw = np.mean(mean_molar_masses)

        if quench_co2_off_co:
            if p_quench is not None:
                p_quench_co2 = kzz_to_co2_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem, log10_metallicities)
                # in between p_quench_co and p_quench_co2, use Keq, then past p_quench_co2, fix co2
                if p_quench_co2 is not None:
                    quenchish_idx = np.logical_and(pressures <= p_quench, pressures <= p_quench_co2)
                else:
                    quenchish_idx = pressures <= p_quench
                mf_h2 = mass_fractions['H2'][quenchish_idx]
                mf_co = mass_fractions['CO'][quenchish_idx]
                mf_h2o = mass_fractions['H2O'][quenchish_idx]
                Keq = 18.3*np.exp((-2376/temperature[quenchish_idx]) - ((932/temperature[quenchish_idx])**2))
                mass_fractions['CO2'][quenchish_idx] = (mf_co * mf_h2o)/(mf_h2 * Keq)
                if p_quench_co2 is not None:
                    quench_idx = np.min(
                        (
                            np.searchsorted(pressures, p_quench_co2),
                            pressures.size - 1
                        )
                    )
                    mass_fractions['CO2'][pressures < p_quench_co2] = \
                        mass_fractions['CO2'][quench_idx]

        if '13CO' in line_species:
            co_total = mass_fractions['CO']
            c_iso_ratio = params['C_iso']
            mass_fractions['13CO'] = co_total/c_iso_ratio
            mass_fractions['12CO'] = co_total-mass_fractions['13CO']

        cloud_particle_radius_distribution_std = params['sigma_lnorm']
        log_kzz_cloud = params['log_kzz_cloud']
        eddy_diffusion_coefficients = np.ones_like(temperature)*1e1**log_kzz_cloud

        cbases = {}
        cloud_f_sed = {}
        for specie in atmosphere_nirspec.cloud_species:
            if 'fsed_' + specie in params.keys():
                cloud_f_sed[specie] = params[f'fsed_{specie}']
            else:
                cloud_f_sed[specie] = params['fsed']
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
        nsmfs = copy.copy(mass_fractions)
        ptmfs = copy.copy(mass_fractions)
        for key in line_species:
            nsmfs[key+f'.R{nsmresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            nsmfs.pop(key.split('_')[0].split('-')[0].split(".")[0], None)
            ptmfs[key+f'.R{ptmresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            ptmfs.pop(key.split('_')[0].split('-')[0].split(".")[0], None)


        wavelengths_nirspec, flux_nirspec, additional_returned_quantities = atmosphere_nirspec.calculate_flux(
            temperatures=temperature,
            mass_fractions=nsmfs,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_f_sed=cloud_f_sed,
            cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
            return_contribution=False
        )

        if 'rv' in params.keys():
            # apply RV shift
            radial_velocity = params['rv'] * 1e5
            # rv in km/s -> 1e5 to cm/s, cst.c in cm/s, wlen first in cm -> micron by 1e4
        else:
            radial_velocity = 0.0
        wavelengths_nirspec *= np.sqrt((1 + radial_velocity/cst.c)/(1- radial_velocity/cst.c))

        phot_wavels = []
        phot_fluxes = []
        for i in range(len(pnames)):
            wavelengths_phot_i, flux_phot_i, additional_returned_quantities = atmosphere_photometrys[i].calculate_flux(
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

            wavelengths_phot_i *= np.sqrt((1 + radial_velocity/cst.c)/(1- radial_velocity/cst.c))

            integrated_flux, _ = synphot.spectrum_to_flux(wavelengths_phot_i*1e4, flux_phot_i* r2d2 * unit_conv) # careful of output here, bc it gives tuple flux, err

            phot_wavels.append(np.nanmean(wavelengths_phot_i*1e4))
            phot_fluxes.append(integrated_flux)


        wavelengths = [wavelengths_nirspec*1e4, phot_wavels]
        flux = [flux_nirspec * r2d2 * unit_conv, phot_fluxes]

        if return_extras:
            if plot:
                wavelengths_plot, flux_plot, additional_returned_quantities = atmosphere_plot.calculate_flux(
                    temperatures=temperature,
                    mass_fractions=nsmfs,
                    mean_molar_masses=mean_molar_masses,
                    reference_gravity=reference_gravity,
                    eddy_diffusion_coefficients=eddy_diffusion_coefficients,
                    cloud_f_sed=cloud_f_sed,
                    cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
                    return_contribution=True
                )
                return wavelengths, flux, wavelengths_plot*1e4, flux_plot* r2d2 * unit_conv, pressures, temperature, mass_fractions, additional_returned_quantities['emission_contribution']
            else:
                raise ValueError("Can't return extras if plot is not true")
        else:
            return wavelengths, flux

    if rank==0:
        param_dict = default_params
        global_params = ['plx', 'C/O', 'Fe/H', 'C_iso', 'e_hat']

        params_bd1 = {}
        params_bd2 = {}
        for key in param_dict.keys():
            if key in global_params:
                params_bd1[key] = param_dict[key]
                params_bd2[key] = param_dict[key]
            else:
                if "_A" in key:
                    key_a = key.split("_A")[0]
                    params_bd1[key_a] = param_dict[key]
                elif "_B" in key:
                    key_b = key.split("_B")[0]
                    params_bd2[key_b] = param_dict[key]
        if 'logg' not in params_bd1.keys():
            if 'M_tot' and 'M_ratio' in param_dict.keys():
                # a over b, so A = tot-B
                M_b = param_dict['M_tot']*param_dict['M_ratio']
                M_a = param_dict['M_tot']-M_b
                params_bd1['mass'] = M_a
                params_bd2['mass'] = M_b
        
        print("params bd1 ")
        print(params_bd1)
        print("params bd2 ")
        print(params_bd2)
    
        t_start = time.time()
        test_w, test_f_1 = spectrum_generator(params_bd1)
        t_end = time.time()
        print('First spectrum Generation time: {:.1f}s'.format(t_end - t_start))
        test_w, test_f_2 = spectrum_generator(params_bd2)

        test_f = test_f_1 + test_f_2
        
        t_start = time.time()
        ln = likelihood(default_params)
        t_end = time.time()
        print('Likelihood time: {:.1f}s'.format(t_end - t_start))
    
        g_test_f = filter_spectrum_with_spline(nsw,spectres(nsw, test_w[0], test_f[0]),x_nodes=x_nodes)
        plt.figure()
        plt.errorbar(nsw, nsf, yerr=nsfe, label='nirspec', color='k', ls='none', marker='.', zorder=10)
        plt.plot(nsw, g_test_f, color='red')
        plt.errorbar(test_w[1], pfs, yerr=pfes, marker='o', color='k', label='photometry')
        plt.errorbar(test_w[1], test_f[1], marker='s', color='red')
        plt.legend()
        plt.savefig(output_dir+'test_alldata_generation.png')

    prior = Prior()
    
    mu_mass = 105
    sigma_mass = 25
    # mu_mass = 3.75
    # sigma_mass = 0.5
    a_mass, b_mass = (1 - mu_mass) / sigma_mass, (200.0 - mu_mass) / sigma_mass
    prior.add_parameter('M_tot', dist=truncnorm(a_mass, b_mass, mu_mass, scale=sigma_mass))
    prior.add_parameter('M_ratio', dist=(0.05, 0.5))
    
    prior.add_parameter('plx', dist=norm(loc=37.561, scale=0.025)) # HD 47127 gaia

    mu_radius = 1.0
    sigma_radius = 0.1
    a_radius, b_radius = (0.75 - mu_radius) / sigma_radius, (2.0 - mu_radius) / sigma_radius
    prior.add_parameter('R_pl_A', 
                        # dist=(0.5, 2.0)
                        dist=truncnorm(a_radius, b_radius, loc=mu_radius, scale=sigma_radius)
                        )
    
    prior.add_parameter('R_pl_B', 
                        # dist=(0.5, 2.0)
                        dist=truncnorm(a_radius, b_radius, loc=mu_radius, scale=sigma_radius)
                        )
    
    prior.add_parameter('C/O', dist=(0.1, 1.0))
    prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    prior.add_parameter('C_iso', dist=(1,150))

    # prior.add_parameter('e_hat', dist=loguniform(1, 1e2))


    # A params
    # prior.add_parameter('T3_A', dist=(0, 1))
    # prior.add_parameter('T2_A', dist=(0, 1))
    # prior.add_parameter('T1_A', dist=(0, 1))
    prior.add_parameter('T_int_A', dist=(200, 1000))
    prior.add_parameter('log_kappa_IR_A', dist=(-6, 8))
    prior.add_parameter('gamma_A', dist=(10**-6, 10**8))

    prior.add_parameter('fsed_A', dist=(0.01, 10))
    prior.add_parameter('sigma_lnorm_A', dist=(1.005, 3))
    prior.add_parameter('log_kzz_cloud_A', dist=(4, 14))
    prior.add_parameter('log_kzz_chem_A', dist=(-5, 25))

    prior.add_parameter('rv_A', dist=(-100, 100))

    # B params
    # prior.add_parameter('T3_B', dist=(0, 1))
    # prior.add_parameter('T2_B', dist=(0, 1))
    # prior.add_parameter('T1_B', dist=(0, 1))
    prior.add_parameter('T_int_B', dist=(100, 500))
    prior.add_parameter('log_kappa_IR_B', dist=(-6, 8))
    prior.add_parameter('gamma_B', dist=(10**-6, 10**8))

    prior.add_parameter('fsed_B', dist=(0.01, 10))
    prior.add_parameter('sigma_lnorm_B', dist=(1.005, 3))
    prior.add_parameter('log_kzz_cloud_B', dist=(4, 14))
    prior.add_parameter('log_kzz_chem_B', dist=(-5, 25))

    prior.add_parameter('rv_B', dist=(-100, 100))


    def prior_pmn(cube, ndim, nparam):
        prior.unit_to_physical(np.ctypeslib.as_array(cube, shape=(ndim,)))
    
    def likelihood_pmn(cube, ndim, nparam):
        param_dict = dict(zip(prior.keys, prior.unit_to_physical(np.ctypeslib.as_array(cube, shape=(ndim,)))))
        return likelihood(param_dict)


    if not pmn:

        # run the sampler!
        if mpied:
            print(f'starting pool with {size} processes')
            comm.Barrier()
            with MPIPool() as pool:
        
                sampler = Sampler(prior, likelihood,
                                n_live=n_live,
                                filepath=checkpoint_file,
                                pool=pool,
                                n_networks=16,
                                resume=resume
                                )
                t_start = time.time()
                sampler.run(f_live=f_live, # default is 0.01, fract of evidence in live set before termination 
                            discard_exploration=discard_exploration, # true for publication ready? fully unbiased
                            verbose=True)
                t_end = time.time()
        else:
            print(f'starting pool with {os.cpu_count()} cores')
            with mp.Pool(os.cpu_count()) as pool:
        
                sampler = Sampler(prior, likelihood,
                                n_live=n_live,
                                filepath=checkpoint_file,
                                pool=pool,
                                n_networks=16,
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
            points[np.isnan(points)] = 0.0
            corndog = corner.corner(
                points, weights=np.exp(log_w), 
                bins=20, labels=prior.keys, color='dodgerblue',
                plot_datapoints=False,
                range=np.repeat(0.999, len(prior.keys))
            )
            plt.savefig(output_dir+f'cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
            print('log Z: {:.2f}'.format(sampler.log_z))
        
            best = points[np.where(log_l==np.nanmax(log_l))][0]
        
    else:
        import pymultinest
        
        comm.Barrier()
        pymultinest.run(
            likelihood_pmn,
            prior_pmn,
            len(prior.keys),
            outputfiles_basename=output_dir,
            resume=resume,
            n_live_points=n_live,
            # **kwargs_multinest,
        )
        if rank==0:
            # Create the Analyzer object
            analyzer = pymultinest.analyse.Analyzer(
                len(prior.keys),
                outputfiles_basename=output_dir,
                verbose=False,
            )

            # Get a dictionary with the ln(Z) and its errors, the
            # individual modes and their parameters quantiles of
            # the parameter posteriors
            sampling_stats = analyzer.get_stats()

            # Nested sampling log-evidence
            ln_z = sampling_stats["nested sampling global log-evidence"]
            ln_z_error = sampling_stats["nested sampling global log-evidence error"]
            print(f"\nlog-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}")

            # Nested importance sampling log-evidence
            imp_ln_z = sampling_stats["nested importance sampling global log-evidence"]
            imp_ln_z_error = sampling_stats[
                "nested importance sampling global log-evidence error"
            ]
            print(
                "log-evidence (importance sampling) = "
                f"{imp_ln_z:.2f} +/- {imp_ln_z_error:.2f}"
            )

            # Get the sample with the maximum likelihood
            best = analyzer.get_best_fit()
            max_lnlike = best["log_likelihood"]
            best = prior.unit_to_physical(np.reshape(best['parameters'], (len(prior.keys),)))

            # Get the posterior samples
            post_samples = analyzer.get_equal_weighted_posterior()

            # Samples and ln(L)
            ln_prob = post_samples[:, -1]
            samples = post_samples[:, :-1]
            samples = prior.unit_to_physical(samples)
            corner.corner(
                samples,
                bins=20, labels=prior.keys, color='dodgerblue',
                plot_datapoints=False,
                range=np.repeat(0.999, len(prior.keys))
            )
            plt.savefig(output_dir+f'cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

    
    print('found best fit parameters:')
    print(best)

    best_params = {}
    for i,param in enumerate(prior.keys):
        best_params[param] = best[i]
        
    if mpied:
        if rank != 0:
            plot = False

    if plot:
        
        global_params = ['plx', 'C/O', 'Fe/H', 'C_iso', 'e_hat']

        params_bd1 = {}
        params_bd2 = {}
        for key in best_params.keys():
            if key in global_params:
                params_bd1[key] = best_params[key]
                params_bd2[key] = best_params[key]
            else:
                if "_A" in key:
                    key_a = key.split("_A")[0]
                    params_bd1[key_a] = best_params[key]
                elif "_B" in key:
                    key_b = key.split("_B")[0]
                    params_bd2[key_b] = best_params[key]
        if 'logg' not in params_bd1.keys():
            if 'M_tot' and 'M_ratio' in best_params.keys():
                # a over b, so A = tot-B
                M_b = best_params['M_tot']*best_params['M_ratio']
                M_a = best_params['M_tot']-M_b
                params_bd1['mass'] = M_a
                params_bd2['mass'] = M_b
        
        print("params bd1 ")
        print(params_bd1)
        print("params bd2 ")
        print(params_bd2)
    
        test_w, test_f_1, _, _, p, t1, mfs1, contribution1 = spectrum_generator(params_bd1, quench_co2_off_co=True, return_extras=True)

        test_w, test_f_2, _, _, p2, t2, mfs2, contribution2 = spectrum_generator(params_bd2, quench_co2_off_co=True, return_extras=True)

        test_f = test_f_1 + test_f_2

        # resample to wl grid
        # rb_f_i = spectres(nsw, w_i[0], cv_f_i) # TODO: use frebin instead of spectres
        rb_f_i = frebin.rebin_spectrum(test_w[0], test_f[0], nsw) # input w, input f, output w
        # print('rebinned spectrum')

        # nirspec_resolution_array = np.interp(w_i[0], nsw, nirspec_resolution)
        # convolve to resolution # TODO: use fortran convolve
        # cv_f_i = convolve(w_i[0], f_i[0], nirspec_resolution_array)
        cv_f_i = fconvolve.variable_width_convolution(nsw, rb_f_i, nirspec_resolution) # input w, input f, res array
        # print('convolved spectrum')
        
        # subtract continuum
        frb_f_i = filter_spectrum_with_spline(nsw,cv_f_i,x_nodes=x_nodes)
        # print('continuum subtracted spectrum')
        
        plt.figure()
        # plt.errorbar(nsw, nsf, yerr=nsfe, label='nirspec', color='k', ls='none')
        # plt.errorbar(test_w[1], test_f[1], marker='s', color='red')
        
        plt.plot(test_w[0], test_f_1[0], label=ln, color='red')
        # plt.errorbar(test_w[1], pfs, yerr=pfes, marker='o', color='k', label='photometry')

        plt.plot(test_w[0], test_f_2[0], label=ln, color='blue')

        # plt.plot(nsw, frb_f_i, label=ln, color='green')

        plt.legend()
        plt.savefig(output_dir+f'compare_twocol_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

        fig, axd = plt.subplot_mosaic(
            """
            A
            B
            C
            D
            E
            F
            G
            H
            I
            """,
            figsize=(12,12)
        )

        axd['A'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none', label='data')
        axd['A'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['A'].set_title('HD 33632 B highpass spectra G395H, pRT diseq. chem.')
        axd['A'].legend(ncol=2, frameon=False, loc='upper right')


        axd['B'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['B'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['B'].set_xlim(2.9,3.2)

        axd['C'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['C'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['C'].set_xlim(3.2,3.5)

        axd['D'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['D'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['D'].set_xlim(3.5,3.8)
        axd['D'].set_ylim(-0.5e-16,0.5e-16)

        axd['E'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['E'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['E'].set_xlim(3.8,4.1)
        axd['E'].set_ylim(-0.5e-16,0.5e-16)

        axd['F'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['F'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['F'].set_xlim(4.1,4.4)
        axd['F'].set_ylim(-0.5e-16,0.5e-16)

        axd['G'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['G'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['G'].set_xlim(4.4,4.7)
        axd['G'].set_ylim(-0.5e-16,0.5e-16)

        axd['H'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['H'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['H'].set_xlim(4.7,5.0)
        axd['H'].set_ylim(-0.5e-16,0.5e-16)

        axd['I'].errorbar(nsw, nsf, yerr=nsfe, marker='', color='k', ls='none')
        axd['I'].plot(nsw, frb_f_i, ls='-', label='model', color='cornflowerblue', alpha=0.5)
        axd['I'].set_xlim(5.0,5.3)
        axd['I'].set_ylim(-0.5e-16,0.5e-16)

        # axd[''].set_xlim(2.8,4.1)
        # axd[''].set_ylim(-2e-17,2e-17)
        axd['E'].set_ylabel('$\mathrm{W}/\mathrm{m}^2/\mathrm{\mu}$m')
        axd['I'].set_xlabel('$\lambda$ [$\mathrm{\mu}$m]')

        for key in list(axd.keys()):
            # if key not in ['I']:
            #     axd[key].tick_params(labelbottom=False)
            axd[key].get_yaxis().get_major_formatter().set_useOffset(False)

        plt.tight_layout()
        plt.savefig(output_dir+f'prt3_best_{retrieval_name}.pdf', dpi=1000, bbox_inches='tight')


        plt.figure()
        plt.hist(nsf-frb_f_i)
        plt.savefig(output_dir+f'prt3_best_residual_hist_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
        
        # plot p-T profile
        plt.figure()
        plt.plot(t1, p, color='r')
        plt.plot(t2, p2, color='b')
        plt.yscale('log')
        plt.ylim(1e3, 1e-6)
        plt.savefig(output_dir+f'pt_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
        
        # plot abundances
        # fig, ax = plt.subplots()
        # i = 0
        # for key in list(mfs.keys()):
        #     if key in ['H2', 'H2O', 'CO', 'CH4', 'CO2', 'MgSiO3(s)_crystalline__DHS']:
        #         ax.plot(mfs[key], p, label=key)
        
        # mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
        #     co_ratios=best_params['C/O']* np.ones_like(p),
        #     log10_metallicities=best_params['Fe/H']* np.ones_like(p),
        #     temperatures=t,
        #     pressures=p,
        #     full=True
        # )  
        # planet_radius = best_params['R_pl']* cst.r_jup_mean
        # reference_gravity = (cst.G*best_params['mass']*cst.m_jup)/(planet_radius**2)
        # co_q = kzz_to_co_pquench(t, p, mean_molar_masses, reference_gravity, best_params['log_kzz_chem'], best_params['Fe/H'])
        # mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
        #     co_ratios=best_params['C/O']* np.ones_like(p),
        #     log10_metallicities=best_params['Fe/H']* np.ones_like(p),
        #     temperatures=t,
        #     pressures=p,
        #     carbon_pressure_quench=co_q,
        #     full=True
        # )
        # co2_q = kzz_to_co2_pquench(t, p, mean_molar_masses, reference_gravity, best_params['log_kzz_chem'], best_params['Fe/H'])
        # ax.hlines(co_q, 1e-8, 9e-1, ls='--', color='black', label='P_q, CO')
        # ax.hlines(co2_q, 1e-8, 9e-1, ls='--', color='gray', label='P_q, CO2')
        # ax.legend(bbox_to_anchor=(1.0, 0.75), ncol=1, fancybox=True, shadow=True)
        # ax.set_xlim(1e-8, 9e-1)
        # ax.set_ylim(1e3, 1e-6)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_title('kzz = '+str(round(best_params['log_kzz_chem'], 2)))
        # plt.savefig(output_dir+f'abundance_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
        
        
        # plot contribution function
        
        # Normalization
        index = (contribution2 < 1e-16) & np.isnan(contribution2)
        contribution2[index] = 1e-16

        pressure_weights = np.diff(np.log10(p))
        weights = np.ones_like(p)
        weights[:-1] = pressure_weights
        weights[-1] = weights[-2]
        weights = weights / np.sum(weights)
        weights = weights.reshape(len(weights), 1)

        x, y = np.meshgrid(test_w[0], p)

        fig, ax = plt.subplots()
        
        plot_cont = contribution2 / weights
        label = "Weighted Flux"

        im = ax.contourf(x,
                            y,
                            plot_cont,
                            30, # n contour levels
                            cmap='magma')
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_ylabel("Pressure [bar]")
        ax.set_yscale("log")
        ax.set_ylim(p[-1] * 1.03, p[0] / 1.03)
        plt.colorbar(im, ax=ax, label=label)
        plt.savefig(output_dir+f'contribution1_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

        # Normalization
        index = (contribution1 < 1e-16) & np.isnan(contribution1)
        contribution1[index] = 1e-16

        pressure_weights = np.diff(np.log10(p))
        weights = np.ones_like(p)
        weights[:-1] = pressure_weights
        weights[-1] = weights[-2]
        weights = weights / np.sum(weights)
        weights = weights.reshape(len(weights), 1)

        x, y = np.meshgrid(test_w[0], p)

        fig, ax = plt.subplots()
        
        plot_cont = contribution1 / weights
        label = "Weighted Flux"

        im = ax.contourf(x,
                            y,
                            plot_cont,
                            30, # n contour levels
                            cmap='magma')
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_ylabel("Pressure [bar]")
        ax.set_yscale("log")
        ax.set_ylim(p[-1] * 1.03, p[0] / 1.03)
        plt.colorbar(im, ax=ax, label=label)
        plt.savefig(output_dir+f'contribution2_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

