import os

import numpy

import park
from park.core.client import JobProxy
from park.core.client import Request

#=============Service Request=====================

class CalcService(Request):
    filepathname = os.path.abspath("..")
    service = os.path.join(filepathname,"csection_service.CrossSectionService")
    version = "0.1"
    require = ["numpy","sympy","scipy","multiprocessing","periodictable"]
    def __init__(self, func, rad, wt, tau, pathname, addargs):
        self.func = func
        self.rad = rad
        self.wt = wt
        self.tau = tau
        self.path = pathname
        self.addargs = addargs

#======Service Proxies===================

class CalcProxy(JobProxy):
    def wait(self):
        print "waiting for results from job"
        JobProxy.wait(self)
        return
    def kill(self):
        print 'killing job'
        JobProxy.kill(self)
    pass

#============ Wrapping the service request==========

@park.core.client.service_wrapper
def calcservice(*args, **kw):
    return CalcService(*args, **kw)