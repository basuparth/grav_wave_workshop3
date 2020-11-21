#!/usr/bin/env python
# coding: utf-8

# <img style="float: left;padding: 1.3em" src="https://indico.in2p3.fr/event/18313/logo-786578160.png">  
# 
# #  Gravitational Wave Open Data Workshop #3
# 
# 
# #### Tutorial 2.5:  Parameter estimation for compact object mergers -- Using and interpreting posterior samples
# 
# This is a simple demonstration to loading and viewing data released in associaton with the publication titled __GWTC-1: A Gravitational-Wave Transient Catalog of Compact Binary Mergers Observed by LIGO and Virgo during the First and Second Observing Runs__ avaliable through [DCC](https://dcc.ligo.org/LIGO-P1800307/public) and [arXiv](https://arxiv.org/abs/1811.12907). This should lead to discussion and interpretation.
# 
# The data used in these tutorials will be downloaded from the public DCC page [LIGO-P1800370](https://dcc.ligo.org/LIGO-P1800370/public).
# 
# [Click this link to view this tutorial in Google Colaboratory](https://colab.research.google.com/github/gw-odw/odw-2019/blob/master/Day_2/Tuto_2.5_Parameter_estimation_for_compact_object_mergers.ipynb)

# ## Installation (execute only if running on a cloud platform!)¶

# In[1]:


# -- Use the following line for google colab
#! pip install -q 'corner==2.0.1'


# **Important**: With Google Colab, you may need to restart the runtime after running the cell above.

# ## Initialization

# In[2]:


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import corner


# ## Get the data
# 
# Selecting the event, let's pick GW150914.

# In[3]:


label = 'GW150914'

# if you do not have wget installed, simply download manually 
# https://dcc.ligo.org/LIGO-P1800370/public/GW150914_GWTC-1.hdf5 
# from your browser
get_ipython().system(' wget https://dcc.ligo.org/LIGO-P1800370/public/{label}_GWTC-1.hdf5')


# In[4]:


posterior_file = './'+label+'_GWTC-1.hdf5'
posterior = h5py.File(posterior_file, 'r')


# ### Looking into the file structure

# In[5]:


print('This file contains four datasets: ',posterior.keys())


# This data file contains several datasets, two using separate models for the gravitaional waveform (`IMRPhenomPv2` and `SEOBNRv3` respectively, see the [paper](https://dcc.ligo.org/LIGO-P1800307) for more details). 
# 
# It also contiains a joint dataset, combining equal numbers of samples from each individual model, these datasets are what is shown in the [paper](https://dcc.ligo.org/LIGO-P1800307). 
# 
# Finally, there is a dataset containing samples drawn from the prior used for the analyses.

# In[6]:


print(posterior['Overall_posterior'].dtype.names)


# Here are some brief descriptions of these parameters and their uses:
# 
#  * `luminosity_distance_Mpc`: luminosity distance [Mpc]
# 
#  * `m1_detector_frame_Msun`: primary (larger) black hole mass (detector frame) [solar mass]
# 
#  * `m2_detector_frame_Msun`: secondary (smaller) black hole mass (detector frame) [solar mass]
# 
#  * `right_ascension`, `declination`: right ascension and declination of the source [rad].
# 
#  * `costheta_jn`: cosine of the angle between line of sight and total angular momentum vector of system.
# 
#  * `spin1`, `costilt1`: primary (larger) black hole spin magnitude (dimensionless) and cosine of the zenith angle between the spin and the orbital angular momentum vector of system.
# 
#  * `spin2`, `costilt2`: secondary (smaller) black hole spin magnitude (dimensionless) and cosine of the zenith angle between the spin and the orbital angular momentum vector of system.
# 
# A convenient (and pretty) way to load up this array of samples is to use [pandas](https://pandas.pydata.org/):

# In[7]:


samples=pd.DataFrame.from_records(np.array(posterior['Overall_posterior']))


# In[8]:


samples


# Those are all the samples stored in the `Overall` dataset. 
# 
# ### Plotting
# 
# We can plot all of them with, for instance, the [corner](https://corner.readthedocs.io/en/latest/) package:

# In[9]:


corner.corner(samples,labels=['costhetajn',
                                'distance [Mpc]',
                                'ra',
                                'dec',
                                'mass1 [Msun]',
                                'mass2 [Msun]',
                                'spin1',
                                'spin2',
                                'costilt1',
                                'costilt2']);


# Each one and two dimentional histogram are *marginalised* probabilby density functions. We can manualy select one parameter, say `luminosity distance`, and plot the four different marginalised distributions:

# In[10]:


plt.hist(posterior['prior']['luminosity_distance_Mpc'], bins = 100, label='prior', alpha=0.8, density=True)
plt.hist(posterior['IMRPhenomPv2_posterior']['luminosity_distance_Mpc'], bins = 100, label='IMRPhenomPv2 posterior', alpha=0.8, density=True)
plt.hist(posterior['SEOBNRv3_posterior']['luminosity_distance_Mpc'], bins = 100, label='SEOBNRv3 posterior', alpha=0.8, density=True)
plt.hist(posterior['Overall_posterior']['luminosity_distance_Mpc'], bins = 100, label='Overall posterior', alpha=0.8, density=True)
plt.xlabel(r'$D_L (Mpc)$')
plt.ylabel('Probability Density Function')
plt.legend()
plt.show()


# ### Computing new quantities
# 
# The masses given are the ones seens by the detector, in the "detector frame". To get the masses of the source black holes, we need to correct for the gravitational-wave redshifting. This forces us to assume a cosmology:

# In[11]:


import astropy.units as u
from astropy.cosmology import Planck15, z_at_value


# We now compute the redshift value for all the samples (using only their distance value). See [astropy.cosmology](http://docs.astropy.org/en/stable/api/astropy.cosmology.z_at_value.html) for implementation details, in particular how to make the following more efficient:

# In[12]:


z = np.array([z_at_value(Planck15.luminosity_distance, dist * u.Mpc) for dist in samples['luminosity_distance_Mpc']])


# In[13]:


samples['m1_source_frame_Msun']=samples['m1_detector_frame_Msun']/(1.0+z)
samples['m2_source_frame_Msun']=samples['m2_detector_frame_Msun']/(1.0+z)
samples['redshift']=z


# And we can plot the marginalised probability density functions:

# In[14]:


corner.corner(samples[['m1_source_frame_Msun','m2_source_frame_Msun','redshift']],labels=['m1 (source)',
                                                                                          'm2 (source)',
                                                                                          'z']);


# ## Calculating credible intervals
# Let's see how we can use bilby to calcuate summary statistics for the posterior like the median and 90% credible level.

# In[15]:


import bilby
# calculate the detector frame chirp mass
mchirp = ((samples['m1_detector_frame_Msun'] * samples['m2_detector_frame_Msun'])**(3./5))/         (samples['m1_detector_frame_Msun'] + samples['m2_detector_frame_Msun'])**(1./5)
# initialize a SampleSummary object to describe the chirp mass posterior samples
chirp_mass_samples_summary = bilby.core.utils.SamplesSummary(samples=mchirp, average='median')
print('The median chirp mass = {} Msun'.format(chirp_mass_samples_summary.median))
print('The 90% confidence interval for the chirp mass is {} - {} Msun'.format(chirp_mass_samples_summary.lower_absolute_credible_interval,
                                                                        chirp_mass_samples_summary.upper_absolute_credible_interval))


# ## Challenge question
# Calculate the posterior for the effective spin, which is the mass-weighted component of the binary spin aligned to the orbital angular momentum. It is given by Eqn. 3 of https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.011001. The z-component of each component spin is defined as $\chi_{1z} = \chi_{1}\cos{\theta_{1}}$. Then initialize a `SamplesSummary` object for the chi_eff posterior and calculate the mean and the lower and upper absolute credible interval. 

# ### $ \chi_{eff} = \dfrac{m_1 \chi_{1z} + m_2 \chi_{2z}}{m_1 + m_2} $
# 
# $ \chi_{eff} = \dfrac{m_1 \chi_{1} \cos \theta_1 + m_2 \chi_{2} \cos \theta_2}{m_1 + m_2} $

# In[26]:


X_det = ((samples['m1_detector_frame_Msun'] * samples['spin1'] * samples['costilt1']) + (samples['m2_detector_frame_Msun'] * samples['spin2'] * samples['costilt2']))/(samples['m1_detector_frame_Msun'] + samples['m2_detector_frame_Msun'])


# In[28]:


X_source = ((samples['m1_source_frame_Msun'] * samples['spin1'] * samples['costilt1']) + (samples['m2_source_frame_Msun'] * samples['spin2'] * samples['costilt2']))/(samples['m1_source_frame_Msun'] + samples['m2_source_frame_Msun'])


# In[31]:


samples['chi_eff_det']=X_det


# In[32]:


samples['chi_eff_source']=X_source


# In[33]:


corner.corner(samples[['m1_source_frame_Msun','m2_source_frame_Msun','chi_eff_source']],labels=['m1 (source)',
                                                                                          'm2 (source)',
                                                                                          'X_source']);


# In[34]:


corner.corner(samples[['m1_source_frame_Msun','m2_source_frame_Msun','chi_eff_det']],labels=['m1 (source)',
                                                                                          'm2 (source)',
                                                                                          'X_det']);


# In[38]:


chi_eff_det_samples_summary = bilby.core.utils.SamplesSummary(samples=X_det, average='median')
print('The median chi_eff_det = {} '.format(chi_eff_det_samples_summary.median))
print('The 90% confidence interval for the chi_eff_det is {} - {} '.format(chi_eff_det_samples_summary.lower_absolute_credible_interval,
                                                                        chi_eff_det_samples_summary.upper_absolute_credible_interval))


# In[39]:


chi_eff_source_samples_summary = bilby.core.utils.SamplesSummary(samples=X_source, average='median')
print('The median chi_eff_source = {} '.format(chi_eff_source_samples_summary.median))
print('The 90% confidence interval for the chi_eff_source is {} - {} '.format(chi_eff_source_samples_summary.lower_absolute_credible_interval,
                                                                        chi_eff_source_samples_summary.upper_absolute_credible_interval))


# In[ ]:




