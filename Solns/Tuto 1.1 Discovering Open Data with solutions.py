#!/usr/bin/env python
# coding: utf-8

# <img style="float: left;padding: 1.3em" src="https://indico.in2p3.fr/event/18313/logo-786578160.png">  
# 
# #  Gravitational Wave Open Data Workshop #3
# 
# 
# #### Tutorial 1.1: Discovering open data from GW observatories
# 
# This notebook describes how to discover what data are available from the [Gravitational-Wave Open Science Center (GWOSC)](https://www.gw-openscience.org).
#     
# [Click this link to view this tutorial in Google Colaboratory](https://colab.research.google.com/github/gw-odw/odw-2020/blob/master/Day_1/Tuto%201.1%20Discovering%20Open%20Data.ipynb)

# ## Software installation  (execute only if running on a cloud platform or haven't done the installation yet!)
# 
# First, we need to install the software, which we do following the instruction in [Software Setup Instructions](https://github.com/gw-odw/odw-2020/blob/master/setup.md):

# In[1]:


# -- Uncomment following line if running in Google Colab
#! pip install -q 'gwosc==0.5.3'


# **Important:** With Google Colab, you may need to restart the runtime after running the cell above.

# In[1]:


#check the version of the package gwosc you are using
import gwosc
print(gwosc.__version__)


# The version you get should be 0.5.3. If it's not, check that you have followed all the steps in [Software Setup Instructions](https://github.com/gw-odw/odw-2020/blob/master/setup.md).

# ## Querying for event information
# 
# The module `gwosc.catalog`  provides tools to search for events in a catalog.
# 
# The module `gwosc.datasets` provides tools for searching for datasets, including full run strain data releases.
# 
# For example, we can search for events in the [GWTC-1 catalog](https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/), the catalog of all events from the O1 and O2 observing runs.  A list of available catalogs can be seen in the [Event Portal](https://gw-openscience.org/eventapi)

# In[2]:


from gwosc.datasets import find_datasets
from gwosc import catalog

#-- Print all the GW events from the GWTC-1 catalog 
gwtc1 = catalog.events('GWTC-1-confident')
print('GWTC-1 events:', gwtc1)
print("")

#-- Print all the large strain data sets from LIGO/Virgo observing runs
runs = find_datasets(type='run')
print('Large data sets:', runs)


# `catalog.events` also accepts a `segment` and `detector` keyword to narrow results based on GPS time and detector:

# In[3]:


#-- Detector and segments keywords limit search result
print(catalog.events('GWTC-1-confident', detector="L1", segment=(1164556817, 1187733618)))


# Using `gwosc.datasets.event_gps`, we can query for the GPS time of a specific event:

# In[4]:


from gwosc.datasets import event_gps
gps = event_gps('GW190425')
print(gps)


# <div class="alert alert-info">All of these times are returned in the GPS time system, which counts the number of seconds that have elapsed since the start of the GPS epoch at midnight (00:00) on January 6th 1980. GWOSC provides a <a href="https://www.gw-openscience.org/gps/">GPS time converter</a> you can use to translate into datetime, or you can use <a href="https://gwpy.github.io/docs/stable/time/"><code>gwpy.time</code></a>.</div>

# We can query for the GPS time interval for an observing run:

# In[5]:


from gwosc.datasets import run_segment
print(run_segment('O1'))


# To see only the confident events in O1:

# In[6]:


O1_events = catalog.events('GWTC-1-confident', segment=run_segment('O1'))
print(O1_events)


# ## Querying for data files
# 
# The `gwosc.locate` module provides a function to find the URLs of data files associated with a given dataset.
# 
# For event datasets, one can get the list of URLs using only the event name:

# In[7]:


from gwosc.locate import get_event_urls
urls = get_event_urls('GW150914')
print(urls)


# By default, this function returns all of the files associated with a given event, which isn't particularly helpful. However, we can can filter on any of these by using keyword arguments, for example to get the URL for the 32-second file for the LIGO-Livingston detector:

# In[8]:


urls = get_event_urls('GW150914', duration=32, detector='L1')
print(urls)


# # Exercises
# 
# Now that you've seen examples of how to query for dataset information using the `gwosc` package, please try and complete the following exercies using that interface:
# 
# - How many months did S6 last?
# - How many GWTC-1-confident events were detected during O1?
# - What file URL contains data for V1 4096 seconds around GW170817?

# In[9]:


from gwosc.datasets import run_segment
print(run_segment('S6'))


# In[12]:


(971622015 - 931035615)/(3600*24*30)


# In[17]:


from gwpy.time import tconvert
t_start = tconvert(931035615)
t_stop = tconvert(971622015)
print('t_start =',t_start)
print('t_stop =',t_stop)


# In[33]:


import datetime
from dateutil.relativedelta import relativedelta

duration = relativedelta(t_stop, t_start)

print ('The S6 run lasted for', duration.years, 'years', duration.months, 'months', duration.days, 'days and', duration.hours, 'hours')


# In[36]:


from gwosc.datasets import run_segment
O1_events = catalog.events('GWTC-1-confident', segment=run_segment('O1'))
print(O1_events)
lO1=len(O1_events)
print('There are', lO1, 'GWTC-1-confident events during the O1 run')


# In[37]:


from gwosc.datasets import run_segment
S6_events = catalog.events('GWTC-1-confident', segment=run_segment('S6'))
print(S6_events)
lS6=len(S6_events)
print('There are', lS6, 'GWTC-1-confident events during the S6 run')


# In[39]:


from gwosc.locate import get_event_urls
urls_GW170817 = get_event_urls('GW170817')
#print(urls_GW170817)
urls_GW170817_V1_4096 = get_event_urls('GW170817', duration=4096, detector='V1')
print('The URL containing the data for V1 for 4096 seconds around GW170817 is',urls_GW170817_V1_4096)


# In[ ]:




