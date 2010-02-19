# Service for csection_calc
# Author: Bill Flynn

# Imports
try:
    import multiprocessing
except ImportError:
    print ' Multiprocessing is not available'

import sys
import os
import time
import Queue
import logging
import traceback

import numpy as np
import sympy as sp


from park.core.client import Service
from cross_section.csection_calc import cs_driver, spherical_averaging

logger = logging.getLogger('cross_section_service')
park.setup_logger(logger=logger, stderr=False)


class CrossSectionService(Service):
    service = 'CrossSection.Service'
#    version = '0.4'
    
    def __init__(self, name=None):
        self.name = name

    def prepare(self, request):
        self.request = request
        self.func = request.func
        self.rad = request.rad
        self.wt = request.wt
        self.tau = request.tau
        self.pathname = request.pathname
        self.addargs = request.addargs
        self.results = []
        
        self.asked = ['rad',len(request.rad),'wt',len(request.wt),'tau',len(request.tau)]
        self.remaining = 0


    def run(self,handler):

        remainder = len(self.tau)
        for t in self.tau:
            vals = spherical_averaging(self.rad, self.wt, t, *self.addargs)
            vals = np.array(vals)
            vals = vas.reshape(len(self.rad),len(self.wt))
            self.results.append(vals)
            
            remainder = remainder - 1
            handler.ready()
        
        self.results = self.results.sum(axis=0)
        
        x = self.rad
        y = self.wt
        z = np.array(self.results)
        
        np.save(os.path.join(self.pathname,r'x_csection_mat.txt'),x)
        np.save(os.path.join(self.pathname,r'y_csection_mat.txt'),y)
        np.save(os.path.join(self.pathname,r'z_csection_mat.txt'),z)
        logger.info("matrices saved with pathname %r" %self.pathname)

    def progress(self):
        return self.asked, self.remaining

    def cleanup(self):
        pass

