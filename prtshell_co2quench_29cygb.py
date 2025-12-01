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
retrieval_name = '29Cygb_shell_kzzchem_onecloud'
output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
discard_exploration = False
f_live = 0.01
n_live = 1000
resume = True
plot = False

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # load data, start species
    time.sleep(rank)
    SpeciesInit()

    data_path = './example_data/'
    sdata = np.loadtxt(data_path+'29cygb_charis_wm2um.dat', delimiter=',')
    hdata = np.loadtxt(data_path+'29cygb_charis_h-band_wm2um.dat', delimiter='\t')
    gdata = fits.getdata(data_path+'29cygb_gravity.fits')
    pdata = data_path+'29cygb_photometry.dat'
    
    sw = sdata[:,0]
    sf = sdata[:,1]
    sfe = sdata[:,2]
    
    hw = hdata[:,0]
    hf = hdata[:,1]
    hfe = hdata[:,2]

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

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron)

    default_params = {
        'R_pl':1.3,
        'plx':37.2539,
        'logg':3.7,

        'T_bottom':7000.,
        'N_layers':6,
        # 'dPT_10':0.02,
        # 'dPT_9':0.02,
        # 'dPT_8':0.02,
        # 'dPT_7':0.06,
        'dPT_6':0.10,
        'dPT_5':0.12,
        'dPT_4':0.15,
        'dPT_3':0.25,
        'dPT_2':0.25,
        'dPT_1':0.25,

        'C/O':0.55,
        'Fe/H':0.0,
        'log_kzz_chem':10,
        'fsed':1,
        'sigma_lnorm':1.5,
        'log_kzz_cloud':10,
        
        'corr_len_ch':-1, # log10 [-3, 0] 
        'corr_amp_ch':0.5 # [0, 1]
    }

    def likelihood(param_dict, debug=False):

        ln = 0

        params = default_params
        for param in param_dict:
            params[param] = param_dict[param]

        w_i, f_i = spectrum_generator(params)

        # do charis low res likelihood:

        rb_f_i = spectres(sw, w_i[0], f_i[0])
        
        chi2 = np.nansum(((sf - rb_f_i)/sfe)**2)
        ln += -chi2/2 - np.nansum(np.log(2*np.pi*sfe**2)/2)
        
        if debug:
            plt.figure()
            plt.plot(sw, rb_f_i)
            
        # do charis h-band likelihood on same model
        
        rb_f_i = spectres(hw, w_i[0], f_i[0])

        if 'corr_len_ch' in param_dict.keys():
            # from Wang et al. 2020, species.fit.fit_model
            wavel_j, wavel_i = np.meshgrid(hw, hw)

            error_j, error_i = np.meshgrid(hfe, hfe)

            corr_len = 10.0 ** param_dict["corr_len_ch"]  # (um)
            corr_amp = param_dict["corr_amp_ch"]

            cov_matrix = (
                corr_amp**2
                * error_i
                * error_j
                * np.exp(-((wavel_i - wavel_j) ** 2) / (2.0 * corr_len**2))
                + (1.0 - corr_amp**2) * np.eye(hw.shape[0]) * error_i**2
            )

            ln_h = (
                (hf - rb_f_i)
                @ np.linalg.inv(cov_matrix)
                @ (hf - rb_f_i)
            )

            ln_h += np.nansum(
                np.log(2.0 * np.pi * hfe**2)
            )

            ln_h *= -0.5

            ln += ln_h
        
        else:
            chi2 = np.nansum(((hf - rb_f_i)/hfe)**2)
            ln += -chi2/2 - np.nansum(np.log(2*np.pi*hfe**2)/2)
        
        if debug:
            plt.figure()
            plt.plot(hw, rb_f_i)

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
            print(ln_h, ln_g, ln_p)

        return ln


    chem = PreCalculatedEquilibriumChemistryTable()
    chem.load()
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
                     # 'Fe(s)_crystalline__DHS'
                    ] # these will be important for clouds

    smresl = '160' # sphere model resolution, R=160 c-k
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
        plotresl = '1000' # gravity model resolution, standard R=1000 c-k
        atmosphere_plot = Radtrans(
            pressures = rtpressures,
            line_species = [i+f'.R{plotresl}' for i in line_species],
            rayleigh_species = rayleigh_species, # why is the sky blue?
            gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
            cloud_species = cloud_species, # these will be important for clouds
            wavelength_boundaries = [0.95, 5.5],
            line_opacity_mode='c-k' # lbl or c-k
        )


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

    def spectrum_generator(params, quench_co2_off_co=True, debug_abund=False, return_extras=False):
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
        if num_layer > 6:
            top_press = -6
        else:
            top_press = -3
        temperature = dtdp_temperature_profile(
            pressures,
            num_layer,
            layer_pt_slopes,
            t_bottom,
            top_of_atmosphere_pressure=top_press,
            bottom_of_atmosphere_pressure=3
        )

        co_ratio = params['C/O']
        feh = params['Fe/H']
        log_kzz_chem = params['log_kzz_chem']

        co_ratios = co_ratio * np.ones_like(pressures)
        log10_metallicities = feh * np.ones_like(pressures)

        mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
            co_ratios=co_ratios,
            log10_metallicities=log10_metallicities,
            temperatures=temperature,
            pressures=pressures,
            full=True
        )

        if debug_abund:
            mf_eqchem = copy.deepcopy(mass_fractions)

        p_quench = kzz_to_co_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem, log10_metallicities)

        if p_quench is not None:

            mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
                co_ratios=co_ratios,
                log10_metallicities=log10_metallicities,
                temperatures=temperature,
                pressures=pressures,
                carbon_pressure_quench=p_quench,
                full=True
            )

        if debug_abund:
            mf_orig = copy.deepcopy(mass_fractions)

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



        if debug_abund:
            mf_list = ['H2', 'H2O', 'CO', 'CH4', 'CO2']
            mf_colors = ['blue', 'red', 'green', 'orange', 'purple']
            fig, ax = plt.subplots()
            i = 0
            for key in list(mass_fractions.keys()):
                if key in mf_list:
                    if key == 'CO2':
                        ax.plot(mass_fractions[key], pressures, label=key, color='k', ls='--')
                        ax.plot(mf_orig[key], pressures, color='k')
                        ax.plot(mf_eqchem[key], pressures, color='k', alpha=0.5)
                        if quench_co2_off_co:
                            if p_quench is not None:
                                ax.hlines(p_quench_co2, 1e-8, 9e-1, ls='--', color='tomato', label='P quench CO2')
                    else:
                        i += 1
                        ax.plot(mass_fractions[key], pressures, label=key, ls='--', color=mf_colors[i])
                        ax.plot(mf_orig[key], pressures, color=mf_colors[i])
                        ax.plot(mf_eqchem[key], pressures, color=mf_colors[i], alpha=0.5)
            if p_quench is not None:
                ax.hlines(p_quench, 1e-8, 9e-1, ls='-', color='tomato', label='P quench CO-CH4-H2O')
            ax.legend()
            ax.set_xlim(1e-8, 9e-1)
            ax.set_ylim(1e3, 1e-6)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.savefig(output_dir+'debug_abund_two.png')

        if '13CO' in atmosphere_sphere.line_species:
            c_iso_ratio = params['C_iso']
            mass_fractions['13CO'] = mass_fractions['CO']/c_iso_ratio
            mass_fractions['12CO'] = mass_fractions['CO']-mass_fractions['13CO']

        sigma_lnorm = params['sigma_lnorm']
        log_kzz_cloud = params['log_kzz_cloud']

        # mmw = params['mmw'] # we get mean_molar_masses from chem.interpolate instead of setting it ourselves
        # mean_molar_masses = mmw * np.ones_like(temperature)
        mmw = np.mean(mean_molar_masses)

        eddy_diffusion_coefficients = np.ones_like(temperature)*1e1**log_kzz_cloud
        cloud_particle_radius_distribution_std = sigma_lnorm

        cbases = {}
        cloud_f_sed = {}
        for specie in atmosphere_sphere.cloud_species:
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

        if return_extras:
            if plot:
                wavelengths_plot, flux_plot, additional_returned_quantities = atmosphere_plot.calculate_flux(
                    temperatures=temperature,
                    mass_fractions=gmfs,
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
        t_start = time.time()
        test_w, test_f = spectrum_generator(default_params)
        t_end = time.time()
        print('First spectrum Generation time: {:.1f}s'.format(t_end - t_start))
        t_start = time.time()
        ln = likelihood(default_params)
        t_end = time.time()
        print('Likelihood time: {:.1f}s'.format(t_end - t_start))
        s_test_f = spectres(sw, test_w[0], test_f[0])
        h_test_f = spectres(hw, test_w[0], test_f[0])
        plt.errorbar(sw, sf, yerr=sfe, label='charis low', marker='.', color='k', ls='none')
        plt.errorbar(hw, hf, yerr=hfe, label='charis h', marker='.', color='k', ls='none')
        g_test_f = spectres(gw, test_w[1], test_f[1])
        plt.errorbar(gw, gf, yerr=gfe, label='gravity', color='k', ls='none')
        plt.plot(sw, s_test_f, label=ln, color='red')
        plt.plot(hw, h_test_f, color='red')
        plt.plot(gw, g_test_f, color='red')
        plt.errorbar(test_w[2], pfs, yerr=pfes, marker='o', color='k', label='photometry')
        plt.errorbar(test_w[2], test_f[2], marker='s', color='red')
        plt.legend()
        plt.savefig(output_dir+'test_alldata_generation.png')

        test_w, test_f_noqco2 = spectrum_generator(default_params, quench_co2_off_co=False, debug_abund=True)
        test_w, test_f_qco2 = spectrum_generator(default_params, quench_co2_off_co=True, debug_abund=True)
        eq_params = copy.deepcopy(default_params)
        eq_params['log_kzz_chem'] = 0.0
        test_w, test_f_alleq = spectrum_generator(eq_params, quench_co2_off_co=False, debug_abund=False)
        plt.figure()
        plt.plot(test_w[2], test_f_noqco2[2], label='quench co2 off quenched co')
        plt.plot(test_w[2], test_f_qco2[2], label='co2 follows chem eq, co quenched (original retrieval)')
        plt.plot(test_w[2], test_f_alleq[2], label='all mol in chem eq')
        plt.xlim(3.5, 5.5)
        plt.ylim(2e-16, 6e-16)
        plt.yscale('log')
        plt.legend()
        plt.savefig(output_dir+'test_co2_quench.png')

    prior = Prior()
    
    mu_radius = 1.3
    sigma_radius = 0.1
    a_radius, b_radius = (0.75 - mu_radius) / sigma_radius, (2.0 - mu_radius) / sigma_radius
    prior.add_parameter('R_pl', dist=truncnorm(a_radius, b_radius, loc=mu_radius, scale=sigma_radius))
    
    mu_mass = 15
    sigma_mass = 5
    # mu_mass = 3.75
    # sigma_mass = 0.5
    a_mass, b_mass = (0.1 - mu_mass) / sigma_mass, (50.0 - mu_mass) / sigma_mass
    prior.add_parameter('mass', dist=truncnorm(a_mass, b_mass, mu_mass, scale=sigma_mass))
    
    # prior.add_parameter('logg', dist=norm(loc=3.7, scale=0.1))
    
    prior.add_parameter('plx', dist=norm(loc=24.5456, scale=0.0911)) # 29 cyg gaia
    # prior.add_parameter('plx', dist=norm(loc=37.2539, scale=0.0195)) # af lep gaia

    prior.add_parameter('T_bottom', dist=(2000, 20000))
    # z23, combo of diff. grids from 10^-3 to 10^3
    prior.add_parameter('dPT_1', dist=norm(loc=0.25, scale=0.025))
    prior.add_parameter('dPT_2', dist=norm(loc=0.25, scale=0.045))
    prior.add_parameter('dPT_3', dist=norm(loc=0.26, scale=0.05))
    prior.add_parameter('dPT_4', dist=norm(loc=0.2, scale=0.05))
    prior.add_parameter('dPT_5', dist=norm(loc=0.12, scale=0.045))
    prior.add_parameter('dPT_6', dist=norm(loc=0.07, scale=0.07))

    # z25, sonora diamondback for 2m1207b
    # prior.add_parameter('dPT_1', dist=(0.05, 0.25)) # diamondback doesn't give P-T below 10^2 so ZJ adopted a uniform prior...
    # prior.add_parameter('dPT_2', dist=norm(loc=0.15, scale=0.01))
    # prior.add_parameter('dPT_3', dist=norm(loc=0.18, scale=0.04))
    # prior.add_parameter('dPT_4', dist=norm(loc=0.21, scale=0.05))
    # prior.add_parameter('dPT_5', dist=norm(loc=0.16, scale=0.06))
    # prior.add_parameter('dPT_6', dist=norm(loc=0.08, scale=0.025))
    # prior.add_parameter('dPT_7', dist=norm(loc=0.06, scale=0.04))
    # prior.add_parameter('dPT_8', dist=(-0.05, 0.1))
    # prior.add_parameter('dPT_9', dist=(-0.05, 0.1))
    # prior.add_parameter('dPT_10', dist=(-0.05, 0.1))
    
    prior.add_parameter('C/O', dist=(0.1, 1.0))
    prior.add_parameter('Fe/H', dist=(-0.5, 2.0))
    prior.add_parameter('log_kzz_chem', dist=(-5, 25))

    # prior.add_parameter('fsed', dist=(0.01, 10))
    prior.add_parameter('fsed_MgSiO3(s)_crystalline__DHS', dist=(1e-4, 10))
    # prior.add_parameter('fsed_Fe(s)_crystalline__DHS', dist=(1e-4, 10))
    
    prior.add_parameter('eq_scaling_MgSiO3(s)_crystalline__DHS', dist=(-3.5, 1))
    # prior.add_parameter('eq_scaling_Fe(s)_crystalline__DHS', dist=(-10, 1))
    
    prior.add_parameter('sigma_lnorm', dist=(1.005, 3))
    prior.add_parameter('log_kzz_cloud', dist=(4, 14))

    prior.add_parameter('corr_len_ch', dist=(-3, 0))
    prior.add_parameter('corr_amp_ch', dist=(0, 1))

    # prior.add_parameter('rv', dist=(-1000, 1000))

    # run the sampler!
    print(f'starting pool with {os.cpu_count()} cores')
    with mp.Pool(os.cpu_count()) as pool:
    # print(f'starting pool with {size} processes')
    # comm.Barrier()
    # with MPIPool() as pool:
        sampler = Sampler(prior, likelihood,
                          n_live=n_live,
                          filepath=checkpoint_file,
                          pool=pool,
                          n_networks=192,
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
    
        if plot:
            test_w, test_f, allw, allf, p, t, mfs, contribution = spectrum_generator(best_params, quench_co2_off_co=True, debug_abund=True, return_extras=True)
            plt.figure()
            s_test_f = spectres(sw, test_w[0], test_f[0])
            plt.errorbar(sw, sf, yerr=sfe, label='charis low', marker='.', color='k', ls='none')
            plt.errorbar(hw, hf, yerr=hfe, label='charis h', marker='.', color='k', ls='none')
            plt.errorbar(gw, gf, yerr=gfe, label='gravity', color='k', ls='none')
            plt.errorbar(test_w[2], test_f[2], marker='s', color='red')
            
            plt.plot(allw, allf, label=ln, color='red')
            plt.errorbar(test_w[2], pfs, yerr=pfes, marker='o', color='k', label='photometry')
            
            plt.legend()
            plt.savefig(output_dir+f'best_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
            
            # plot p-T profile
            plt.figure()
            plt.plot(t, p, color='k')
            plt.yscale('log')
            plt.ylim(1e3, 1e-6)
            plt.savefig(output_dir+f'pt_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
            
            # plot abundances
            fig, ax = plt.subplots()
            i = 0
            for key in list(mfs.keys()):
                if key in ['H2', 'H2O', 'CO', 'CH4', 'CO2', 'MgSiO3(s)_crystalline__DHS']:
                    ax.plot(mfs[key], p, label=key)
            
            mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
                co_ratios=best_params['C/O']* np.ones_like(p),
                log10_metallicities=best_params['Fe/H']* np.ones_like(p),
                temperatures=t,
                pressures=p,
                full=True
            )  
            planet_radius = best_params['R_pl']* cst.r_jup_mean
            reference_gravity = (cst.G*best_params['mass']*cst.m_jup)/(planet_radius**2)
            co_q = kzz_to_co_pquench(t, p, mean_molar_masses, reference_gravity, best_params['log_kzz_chem'], best_params['Fe/H'])
            mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
                co_ratios=best_params['C/O']* np.ones_like(p),
                log10_metallicities=best_params['Fe/H']* np.ones_like(p),
                temperatures=t,
                pressures=p,
                carbon_pressure_quench=co_q,
                full=True
            )
            co2_q = kzz_to_co2_pquench(t, p, mean_molar_masses, reference_gravity, best_params['log_kzz_chem'], best_params['Fe/H'])
            ax.hlines(co_q, 1e-8, 9e-1, ls='--', color='black', label='P_q, CO')
            ax.hlines(co2_q, 1e-8, 9e-1, ls='--', color='gray', label='P_q, CO2')
            ax.legend(bbox_to_anchor=(1.0, 0.75), ncol=1, fancybox=True, shadow=True)
            ax.set_xlim(1e-8, 9e-1)
            ax.set_ylim(1e3, 1e-6)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('kzz='+str(round(best_params['log_kzz_chem'], 2)))
            plt.savefig(output_dir+f'abundance_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')
            
            
            # plot contribution function
            
            # Normalization
            index = (contribution < 1e-16) & np.isnan(contribution)
            contribution[index] = 1e-16
    
            pressure_weights = np.diff(np.log10(p))
            weights = np.ones_like(p)
            weights[:-1] = pressure_weights
            weights[-1] = weights[-2]
            weights = weights / np.sum(weights)
            weights = weights.reshape(len(weights), 1)
    
            x, y = np.meshgrid(allw, p)
    
            fig, ax = plt.subplots()
            
            plot_cont = contribution / weights
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
            plt.savefig(output_dir+f'contribution_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')

