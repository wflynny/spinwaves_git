import sys
import math
import copy
import wx
import numpy as np
from numpy import pi
from mpfit.mpfit import mpfit
import spinwaves.vtkModel.Parameter_Manager as PM
from wx.py.dispatcher import send
from anneal import anneal
from spinwaves.vtkModel.wxGUI.Session import Session
from spinwaves.vtkModel.BondClass import bondListGrid


def fitting(session, spinwave_domain, spinwave_range, spinwave_range_Err, size=3, k = 100, tMin = .001, tMax = 15, tFactor = .95, MCeveryTime = True, recordKeeperCallback = None):

    # Create fitter
    fitter = PM.Fitter(session, spinwave_domain, size, k, tMin, tMax, tFactor, MCeveryTime)
    
    # Helper function for GetResult
    def myfunc(p, fjac=None, y=None, err=None, callback = None):
        if callback:
            callback(p)
        fitter.fit_list = p
        model = fitter.GetResult()
        status = 0
        print 'y:\n', y, '\n\nmodel:\n', model
        #print '\n\ny-model:\n', (y-model)
        #print '\n\nerr:\n', err
        result = (y-model)/err
        print '\n\nresult:\n', result
        return [status, result]

    # Function Keywords
    
    #spinwave_domain is a list of 3D lists:
    #-[(x1,y1,z1),(x2,y2,z2),...]
    #x must be a 1D numpy array:
    #[x1,y1,z1,x2,t2,z2,...]
    #xList = []
    #for point in fit.spinwave_domain:
    #    xList.append(point[0])
    #    yList.append(point[1])
    #    zList.append(point[2])
    #x = np.array(xList)
    #y = myfunc(fit)
    y = spinwave_range
    #errVal = .001
    #err = [errVal]*len(y)
    err = spinwave_range_Err
    #fa = {'x':x, 'y':y, 'err':err}
    fa = {'y':y, 'err':err, 'callback':recordKeeperCallback}
    
    # Set parinfo with the limits and such. Probably don't need
    parbase={'value':0., 'limited':[0,0], 'limits':[0.,0.]}
    parinfo=[]
    p0 = fitter.fit_list
    for i in range(len(p0)):
        parinfo.append(copy.deepcopy(parbase))
    for i in range(len(p0)):
        parinfo[i]['value']=p0[i]
        if(fitter.min_range_list[i] != np.NINF):
            parinfo[i]['limits'][0]=fitter.min_range_list[i]
            parinfo[i]['limited'][0] = 1
        else:
            parinfo[i]['limited'][0] = 0
        if fitter.max_range_list[i] != np.PINF:
            parinfo[i]['limits'][1] = fitter.max_range_list[i]
            parinfo[i]['limited'][1] = 1
        else:
            parinfo[i]['limited'][1] = 0

    # Run mpfit on fitlist with all the jazz.
    print "params: ", p0
    print "parinfo: ", parinfo
    m = mpfit(myfunc, p0, parinfo=parinfo, functkw = fa)
    # Bare Bones Run for when the above (especially keywords x) breaks instantly
    #m = mpfit(myfunc, fit.fitlist)
    
    #Return the parameters
    return (m.status, m.params, m.perror)

def annealFit(session, spinwave_domain, spinwave_range, spinwave_range_Err, size=3, k = 100, tMin = .001, tMax = 15, tFactor = .95, MCeveryTime = True, recordKeeperCallback = None):
    # Create fitter
    fitter = PM.Fitter(session, spinwave_domain, size, k, tMin, tMax, tFactor, MCeveryTime)
    #testFile = open("/home/tom/Desktop/annealOutput1.txt", 'w')
    #testFile = open("C:\\annealOutput1.txt", 'w')
    
    # Helper function for GetResult
    def myfunc(p, y=None, err=None, callback = None):
        """returns Chi Squared to be minimized."""
        if callback:
            callback(p)
            
        #testFile.write('\np: ' + str(p))
        #testFile.flush()
        fitter.fit_list = p
        model = fitter.GetResult()
        #print 'y:\n', y, '\n\nmodel:\n', model
        #for i in range(len(y)):
        #    testFile.write("  p: p %3.5f y %3.5f model %3.5f err %3.5f"%(p,y[i],model[i],err[i]) )
        result = (y-model)/err
        chi_sq = 0
        for entry in result:
            #print "entry: ", entry
            chi_sq += math.pow(entry, 2)
        #print '\n\nresult:\n', result
        #print "\nchi_sq: " + str(chi_sq)
        #testFile.write("\nchi_sq: " + str(chi_sq))
        #testFile.flush()
        return chi_sq

    # Function Keywords
    y = spinwave_range
    err = spinwave_range_Err
    p0 = fitter.fit_list
    #testFile.write("initial p: " + str(p0))
    
    #_______________________ Testing _____________________________
    #START = -10
    #STOP = 10
    #step = .01
    
    #FILE = "C:\\CHISQ_points3.txt"
    
    ##plot
    
    #handle = open(FILE, 'w')
    ##handle.write("data: "+ str(y))
    ##handle.flush()
    #for i in range((STOP-START)/step):
	#x = START + i*step
	#val = myfunc([x], y, err)
	#if i != 0:
	    #handle.write("\n")
	#handle.write(str(x) + " " + str(val))
    #handle.close()
    #sys.exit()
	
	
    #_____________________________________________________________________


    result=anneal(myfunc,p0,args=(y,err, recordKeeperCallback), schedule='simple',lower=fitter.min_range_list,upper=fitter.max_range_list,
                  maxeval=None, maxaccept=None,dwell=50, maxiter=2000, full_output=1)
    
    print "annealing result: ", result
    p = result[0]
    myfunc(p, y, err)#To propogate final value

    
    #----Now Localy optimize with mpfit to fine tune the answer -----------
    def mpfitfunc(p, fjac=None, y=None, err=None, callback = None):
        if callback:
            callback(p)
	#testFile.write("mpfit: " + str(p))
        fitter.fit_list = p
        model = fitter.GetResult()
        status = 0
        print 'y:\n', y, '\n\nmodel:\n', model
        #print '\n\ny-model:\n', (y-model)
        #print '\n\nerr:\n', err
        result = (y-model)/err
        print '\n\nresult:\n', result
        return [status, result]
    
    fa = {'y':y, 'err':err, 'callback':recordKeeperCallback}
    
    # Set parinfo with the limits and such. Probably don't need
    parbase={'value':0., 'limited':[0,0], 'limits':[0.,0.]}
    parinfo=[]
    p0 = fitter.fit_list
    for i in range(len(p0)):
        parinfo.append(copy.deepcopy(parbase))
    for i in range(len(p0)):
        parinfo[i]['value']=p0[i]
        if(fitter.min_range_list[i] != np.NINF):
            parinfo[i]['limits'][0]=fitter.min_range_list[i]
            parinfo[i]['limited'][0] = 1
        else:
            parinfo[i]['limited'][0] = 0
        if fitter.max_range_list[i] != np.PINF:
            parinfo[i]['limits'][1] = fitter.max_range_list[i]
            parinfo[i]['limited'][1] = 1
        else:
            parinfo[i]['limited'][1] = 0

    # Run mpfit on fitlist with all the jazz.
    print "params: ", p0
    print "parinfo: ", parinfo
    m = mpfit(mpfitfunc, p0, parinfo=parinfo, functkw = fa)
    #testFile.write("\nresult: " + str(result))
    #testFile.write("\nm: " + str(m))
    #testFile.close()
    return (m.status, m.params, m.perror)
    #---------------------------------------------------------------------
    
#    testFile.write("\nresult: " + str(result))
#    testFile.close()
#    return p

def readDataFile(fileName):
    vals = np.loadtxt(fname = fileName, comments = '#')
    hklPoints = []
    hklErr = []
    wVals = []
    wErr = []
    print vals
    for row in vals:
        hklPoints.append([row[0], row[2], row[4]])
        hklErr.append([row[1], row[3], row[5]])
        wVals.append(row[6])
        wErr.append(row[7])
    
    print 'hklPoints:\n', hklPoints, '\n\nhklErr::\n', hklErr
    print 'wPoints:\n', wVals
    print 'wErr', wErr
    
    return hklPoints, hklErr, wVals, wErr

def fitFromFile(fileName, session, size = 3 , k = 100, tMin = .001, tMax = 15, tFactor = .95, MCeveryTime = True, recordKeeper = None):
    domain, xErr, w, wErr = readDataFile(fileName)
    return fitting(session, domain, w, wErr, size = size, k = k, tMin = tMin, tMax = tMax, tFactor = tFactor, MCeveryTime = MCeveryTime, recordKeeperCallback = recordKeeper)

def annealFitFromFile(fileName, session, size = 3 , k = 100, tMin = .001, tMax = 15, tFactor = .95, MCeveryTime = True, recordKeeper = None):
    domain, xErr, w, wErr = readDataFile(fileName)
    print "\n\n\n\n\nAnnealing!\n\n\n\n"
    return annealFit(session, domain, w, wErr, size = size, k = k, tMin = tMin, tMax = tMax, tFactor = tFactor, MCeveryTime = MCeveryTime, recordKeeperCallback = recordKeeper)
#Not dealing with domain err
#def propogate_uncertainty(func, point, err, delt_h = 1E-12, delt_k = 1E-12, delt_l = 1E-12):
    #""" Point is of the formL (h,k,l), and err is of the form: (sigma_h, sigma_k, sigma_l).
    #In order to estimate the partial diriviatives o fhte function with respect to h,k, and l, 
    #the function is evaluated at points delt_h, delt_k, and delt_l distance away from the point
    #and the slope at this small interval is used as an estimate of hte dirivative."""
    
    ##first we need to know the partial dirivatives of the function for h,k, and l
    #point1 = func(point[0] - delt_h, point[1], point[2])
    #point2 = func(point[0] + delt_h, point[1], point[2])
    #rise = point2 - point1
    #partial_h = rise/(2*delt_h)
    
    #point1 = func(point[0], point[1] - delt_k, point[2])
    #point2 = func(point[0], point[1] + delt_k, point[2])
    #rise = point2 - point1
    #partial_k = rise/(2*delt_k)
    
    #point1 = func(point[0], point[1], point[2] - delt_l)
    #point2 = func(point[0], point[1], point[2] + delt_l)
    #rise = point2 - point1
    #partial_l = rise/(2*delt_l)
    
    #err = np.sqrt(



class FitPanel(wx.Panel):
    def __init__(self, session, procManager, *args, **kwds):
	self.session = session
	self.procManager = procManager
        # begin wxGlade: FitPanel.__init__
        kwds["style"] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, *args, **kwds)
        self.sizer_3_staticbox = wx.StaticBox(self, -1, "Simulated Annealing")
        self.path_label = wx.StaticText(self, -1, " Data File:  ")
        self.pathCtrl = wx.TextCtrl(self, -1, "")
        self.browse_btn = wx.Button(self, -1, "Browse")
        self.size_label = wx.StaticText(self, -1, " Lattice Size:  ")
        self.sizeCtrl = wx.TextCtrl(self, -1, "3")
        self.fitter_label = wx.StaticText(self, -1, " Fitting Algorithm:  ")
        self.fitter_combo_box = wx.ComboBox(self, -1, choices=["mp_fit local optimizer (fast)          ", "simulated annealing (slow)       "], style=wx.CB_DROPDOWN)
        self.radio_box_1 = wx.RadioBox(self, -1, "Ground State", choices=["Calculate Ground State", "Load Ground State Spins"], majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        self.monteCarlo_checkbox = wx.CheckBox(self, -1, "use simulated annealing on every iteration")
        self.maxT_label = wx.StaticText(self, -1, " Max Temperature:  ")
        self.maxTCtrl = wx.TextCtrl(self, -1, "15")
        self.minTemp_label = wx.StaticText(self, -1, " Min Temperature:  ")
        self.minTCtrl = wx.TextCtrl(self, -1, ".01")
        self.k_label = wx.StaticText(self, -1, " k: ")
        self.kCtrl = wx.TextCtrl(self, -1, "100")
        self.tFact_label = wx.StaticText(self, -1, " Temperature Factor:  ")
        self.tFactCtrl = wx.TextCtrl(self, -1, ".9")
        self.spinPath_label = wx.StaticText(self, -1, "Spin File:  ")
        self.pathCtrl_copy = wx.TextCtrl(self, -1, "")
        self.spinBrowse_btn = wx.Button(self, -1, "Browse")
        self.fit_btn = wx.Button(self, -1, "Fit")

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.OnBrowse, self.browse_btn)
        self.Bind(wx.EVT_RADIOBOX, self.OnRadioBoxChange, self.radio_box_1)
        self.Bind(wx.EVT_BUTTON, self.OnBrowse, self.spinBrowse_btn)
        self.Bind(wx.EVT_BUTTON, self.OnFit, self.fit_btn)
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: FitPanel.__set_properties
        self.pathCtrl.SetMinSize((180, 27))
        self.fitter_combo_box.SetSelection(0)
        self.radio_box_1.SetSelection(0)
        self.spinPath_label.Enable(False)
        self.pathCtrl_copy.SetMinSize((180, 27))
        self.pathCtrl_copy.Enable(False)
        self.spinBrowse_btn.Enable(False)
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: FitPanel.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2_copy = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3 = wx.StaticBoxSizer(self.sizer_3_staticbox, wx.VERTICAL)
        sizer_8 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_9 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2.Add(self.path_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2.Add(self.pathCtrl, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2.Add(self.browse_btn, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 1)
        sizer_6.Add(self.size_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_6.Add(self.sizeCtrl, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_1.Add(sizer_6, 1, 0, 0)
        sizer_9.Add(self.fitter_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_9.Add(self.fitter_combo_box, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_1.Add(sizer_9, 1, wx.EXPAND, 0)
        sizer_1.Add(self.radio_box_1, 0, 0, 0)
        sizer_3.Add(self.monteCarlo_checkbox, 0, 0, 0)
        sizer_4.Add(self.maxT_label, 0, 0, 0)
        sizer_4.Add(self.maxTCtrl, 0, 0, 0)
        sizer_3.Add(sizer_4, 1, wx.EXPAND, 0)
        sizer_5.Add(self.minTemp_label, 0, 0, 0)
        sizer_5.Add(self.minTCtrl, 0, 0, 0)
        sizer_3.Add(sizer_5, 1, wx.EXPAND, 0)
        sizer_7.Add(self.k_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_7.Add(self.kCtrl, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_3.Add(sizer_7, 1, wx.EXPAND, 0)
        sizer_8.Add(self.tFact_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_8.Add(self.tFactCtrl, 0, 0, 0)
        sizer_3.Add(sizer_8, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_3, 3, wx.EXPAND, 0)
        sizer_2_copy.Add(self.spinPath_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2_copy.Add(self.pathCtrl_copy, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2_copy.Add(self.spinBrowse_btn, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_1.Add(sizer_2_copy, 1, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)
        sizer_1.Add((20, 10), 0, 0, 0)
        sizer_1.Add(self.fit_btn, 0, wx.ALIGN_CENTER_HORIZONTAL, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        # end wxGlade

    def OnBrowse(self, event): # wxGlade: FittingFrame.<event_handler>
        #code essentially copied from OnBrowseInteractions in cross section GUI
        confBase = wx.ConfigBase.Create()
        confBase.SetStyle(wx.CONFIG_USE_LOCAL_FILE)
        defaultDir=confBase.Get().GetPath()
        wildcard="files (*.txt)|*.txt|All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self, message="Choose a data file.",
            defaultDir=defaultDir,
            defaultFile="",
            wildcard=wildcard,
            style=wx.OPEN | wx.CHANGE_DIR
            )

        # Show the dialog and retrieve the user response. If it is the OK response,
        # process the data.
        if dlg.ShowModal() == wx.ID_OK:
            # This returns a Python list of files that were selected.
            paths = dlg.GetPaths()
            self.pathCtrl.SetValue(paths[0])


    def OnFit(self, event): # wxGlade: FittingFrame.<event_handler>
        failed, useMonteCarlo, tmax, tmin, tfactor, k, size = self.validate()
        if not failed:
	    useMC =  self.monteCarlo_checkbox.GetValue() 
	    fileName = self.pathCtrl.GetValue() #Not checked right now since most people will browse
	    self.procManager.startFit(self.session, fileName, k, tmin, tmax, size, tfactor, useMC, fitType = self.fitter_combo_box.GetCurrentSelection())
	    #ans = fitFromFile(fileName, self.session, k = k, tMax = tmax, tMin = tmin, size = size, tFactor = tfactor, MCeveryTime = useMC)
	    #print "ans =\n", ans[1]
    
        
    def validate(self):
        """Validates the info entered for tMin, tMax, tFactor, k and lattice size 
        to make sure they are of the right types."""
	useMC = self.radio_box_1.GetSelection()
        tMax = self.maxTCtrl.GetValue()
        tMin = self.minTCtrl.GetValue()
        tFactor = self.tFactCtrl.GetValue()
        steps = self.kCtrl.GetValue()
        size = self.sizeCtrl.GetValue()
        #For now, I will not validate the file path
        #inPath = self.inPathtext.GetValue()#Can check if it is readable
        
        bgColor = "pink"
        failed = False

	if useMC == 0:
		#Validate tMax(must be a float)
		numTmax = None
		try:
		    numTmax = float(tMax)
		    self.maxTCtrl.SetStyle(0, len(tMax), wx.TextAttr(colBack = "white"))
		except:
		    self.maxTCtrl.SetStyle(0, len(tMax), wx.TextAttr(colBack = bgColor))
		    failed = True
		
		#Validate tMin(must be a float)
		numTmin = None
		try:
		    numTmin = float(tMin)
		    self.minTCtrl.SetStyle(0, len(tMin), wx.TextAttr(colBack = "white"))
		except:
		    self.minTCtrl.SetStyle(0, len(tMin), wx.TextAttr(colBack = bgColor))
		    failed = True
		
		#Validate tFactor(must be a float)
		numTfactor = None
		try:
		    numTfactor = float(tFactor)
		    self.tFactCtrl.SetStyle(0, len(tFactor), wx.TextAttr(colBack = "white"))
		except:
		    self.tFactCtrl.SetStyle(0, len(tFactor), wx.TextAttr(colBack = bgColor))
		    failed = True
		    
		#Validate steps(must be an int)
		numSteps = None
		try:
		    numSteps = int(steps)
		    self.kCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = "white"))
		except:
		    self.kCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = bgColor))
		    failed = True
		    
		#Validate size(must be an int)
		numSize = None
		try:
		    numSize = int(size)
		    self.sizeCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = "white"))
		except:
		    self.sizeCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = bgColor))
		    failed = True
		
            
        return failed, useMC, numTmax, numTmin, numTfactor, numSteps, numSize
        

    def OnRadioBoxChange(self, event): # wxGlade: FittingFrame.<event_handler>
	val = self.radio_box_1.GetSelection()
	print val
	if val == 0:
		self.spinPath_label.Enable(False)
		self.pathCtrl_copy.Enable(False)
		self.spinBrowse_btn.Enable(False)

		self.monteCarlo_checkbox.Enable(True)
		self.maxT_label.Enable(True)
		self.maxTCtrl.Enable(True)
		self.minTemp_label.Enable(True)
		self.minTCtrl.Enable(True)
		self.k_label.Enable(True)
		self.kCtrl.Enable(True)
		self.tFact_label.Enable(True)
		self.tFactCtrl.Enable(True)

	else:
		print "Loading a Ground State Spin File is not yet supported!"
		self.radio_box_1.SetSelection(0)
#		self.spinPath_label.Enable(True)
#		self.pathCtrl_copy.Enable(True)
#		self.spinBrowse_btn.Enable(True)
#
#		self.monteCarlo_checkbox.Enable(False)
#		self.maxT_label.Enable(False)
#		self.maxTCtrl.Enable(False)
#		self.minTemp_label.Enable(False)
#		self.minTCtrl.Enable(False)
#		self.k_label.Enable(False)
#		self.kCtrl.Enable(False)
#		self.tFact_label.Enable(False)
#		self.tFactCtrl.Enable(False)

        event.Skip()

# end of class FitPanel



#class FitPanel(wx.Panel):
    #def __init__(self, session, procManager, *args, **kwds):
	#self.session = session
	#self.procManager = procManager
        ## begin wxGlade: FitPanel.__init__
        #kwds["style"] = wx.TAB_TRAVERSAL
        #wx.Panel.__init__(self, *args, **kwds)
        #self.sizer_3_staticbox = wx.StaticBox(self, -1, "Simulated Annealing")
        #self.path_label = wx.StaticText(self, -1, " Data File:  ")
        #self.pathCtrl = wx.TextCtrl(self, -1, "")
        #self.browse_btn = wx.Button(self, -1, "Browse")
        #self.size_label = wx.StaticText(self, -1, " Lattice Size:  ")
        #self.sizeCtrl = wx.TextCtrl(self, -1, "3")
        #self.radio_box_1 = wx.RadioBox(self, -1, "Ground State", choices=["Calculate Ground State", "Load Ground State Spins"], majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        #self.monteCarlo_checkbox = wx.CheckBox(self, -1, "use simulated annealing on every iteration")
        #self.maxT_label = wx.StaticText(self, -1, " Max Temperature:  ")
        #self.maxTCtrl = wx.TextCtrl(self, -1, "15")
        #self.minTemp_label = wx.StaticText(self, -1, " Min Temperature:  ")
        #self.minTCtrl = wx.TextCtrl(self, -1, ".01")
        #self.k_label = wx.StaticText(self, -1, " k: ")
        #self.kCtrl = wx.TextCtrl(self, -1, "100")
        #self.tFact_label = wx.StaticText(self, -1, " Temperature Factor:  ")
        #self.tFactCtrl = wx.TextCtrl(self, -1, ".9")
        #self.spinPath_label = wx.StaticText(self, -1, "Spin File:  ")
        #self.pathCtrl_copy = wx.TextCtrl(self, -1, "")
        #self.spinBrowse_btn = wx.Button(self, -1, "Browse")
        #self.fit_btn = wx.Button(self, -1, "Fit")

        #self.__set_properties()
        #self.__do_layout()

        #self.Bind(wx.EVT_BUTTON, self.OnBrowse, self.browse_btn)
        #self.Bind(wx.EVT_RADIOBOX, self.OnRadioBoxChange, self.radio_box_1)
        #self.Bind(wx.EVT_BUTTON, self.OnBrowse, self.spinBrowse_btn)
        #self.Bind(wx.EVT_BUTTON, self.OnFit, self.fit_btn)
        ## end wxGlade

    #def __set_properties(self):
        ## begin wxGlade: FitPanel.__set_properties
        #self.pathCtrl.SetMinSize((180, 27))
        #self.radio_box_1.SetSelection(0)
        #self.spinPath_label.Enable(False)
        #self.pathCtrl_copy.SetMinSize((180, 27))
        #self.pathCtrl_copy.Enable(False)
        #self.spinBrowse_btn.Enable(False)
        ## end wxGlade

    #def __do_layout(self):
        ## begin wxGlade: FitPanel.__do_layout
        #sizer_1 = wx.BoxSizer(wx.VERTICAL)
        #sizer_2_copy = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_3 = wx.StaticBoxSizer(self.sizer_3_staticbox, wx.VERTICAL)
        #sizer_8 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_2.Add(self.path_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_2.Add(self.pathCtrl, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_2.Add(self.browse_btn, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_1.Add(sizer_2, 1, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 1)
        #sizer_6.Add(self.size_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_6.Add(self.sizeCtrl, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_1.Add(sizer_6, 1, 0, 0)
        #sizer_1.Add(self.radio_box_1, 0, 0, 0)
        #sizer_3.Add(self.monteCarlo_checkbox, 0, 0, 0)
        #sizer_4.Add(self.maxT_label, 0, 0, 0)
        #sizer_4.Add(self.maxTCtrl, 0, 0, 0)
        #sizer_3.Add(sizer_4, 1, wx.EXPAND, 0)
        #sizer_5.Add(self.minTemp_label, 0, 0, 0)
        #sizer_5.Add(self.minTCtrl, 0, 0, 0)
        #sizer_3.Add(sizer_5, 1, wx.EXPAND, 0)
        #sizer_7.Add(self.k_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_7.Add(self.kCtrl, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_3.Add(sizer_7, 1, wx.EXPAND, 0)
        #sizer_8.Add(self.tFact_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_8.Add(self.tFactCtrl, 0, 0, 0)
        #sizer_3.Add(sizer_8, 1, wx.EXPAND, 0)
        #sizer_1.Add(sizer_3, 3, wx.EXPAND, 0)
        #sizer_2_copy.Add(self.spinPath_label, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_2_copy.Add(self.pathCtrl_copy, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_2_copy.Add(self.spinBrowse_btn, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        #sizer_1.Add(sizer_2_copy, 1, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)
        #sizer_1.Add((20, 10), 0, 0, 0)
        #sizer_1.Add(self.fit_btn, 0, wx.ALIGN_CENTER_HORIZONTAL, 0)
        #self.SetSizer(sizer_1)
        #sizer_1.Fit(self)
        ## end wxGlade

    #def OnBrowse(self, event): # wxGlade: FittingFrame.<event_handler>
        ##code essentially copied from OnBrowseInteractions in cross section GUI
        #confBase = wx.ConfigBase.Create()
        #confBase.SetStyle(wx.CONFIG_USE_LOCAL_FILE)
        #defaultDir=confBase.Get().GetPath()
        #wildcard="files (*.txt)|*.txt|All files (*.*)|*.*"
        #dlg = wx.FileDialog(
            #self, message="Choose a data file.",
            #defaultDir=defaultDir,
            #defaultFile="",
            #wildcard=wildcard,
            #style=wx.OPEN | wx.CHANGE_DIR
            #)

        ## Show the dialog and retrieve the user response. If it is the OK response,
        ## process the data.
        #if dlg.ShowModal() == wx.ID_OK:
            ## This returns a Python list of files that were selected.
            #paths = dlg.GetPaths()
            #self.pathCtrl.SetValue(paths[0])


    #def OnFit(self, event): # wxGlade: FittingFrame.<event_handler>
        #failed, useMonteCarlo, tmax, tmin, tfactor, k, size = self.validate()
        #if not failed:
	    #useMC =  self.monteCarlo_checkbox.GetValue() 
	    #fileName = self.pathCtrl.GetValue() #Not checked right now since most people will browse
	    #self.procManager.startFit(self.session, fileName, k, tmin, tmax, size, tfactor, useMC)
	    ##ans = fitFromFile(fileName, self.session, k = k, tMax = tmax, tMin = tmin, size = size, tFactor = tfactor, MCeveryTime = useMC)
	    ##print "ans =\n", ans[1]
    
        
    #def validate(self):
        #"""Validates the info entered for tMin, tMax, tFactor, k and lattice size 
        #to make sure they are of the right types."""
	#useMC = self.radio_box_1.GetSelection()
        #tMax = self.maxTCtrl.GetValue()
        #tMin = self.minTCtrl.GetValue()
        #tFactor = self.tFactCtrl.GetValue()
        #steps = self.kCtrl.GetValue()
        #size = self.sizeCtrl.GetValue()
        ##For now, I will not validate the file path
        ##inPath = self.inPathtext.GetValue()#Can check if it is readable
        
        #bgColor = "pink"
        #failed = False

	#if useMC == 0:
		##Validate tMax(must be a float)
		#numTmax = None
		#try:
		    #numTmax = float(tMax)
		    #self.maxTCtrl.SetStyle(0, len(tMax), wx.TextAttr(colBack = "white"))
		#except:
		    #self.maxTCtrl.SetStyle(0, len(tMax), wx.TextAttr(colBack = bgColor))
		    #failed = True
		
		##Validate tMin(must be a float)
		#numTmin = None
		#try:
		    #numTmin = float(tMin)
		    #self.minTCtrl.SetStyle(0, len(tMin), wx.TextAttr(colBack = "white"))
		#except:
		    #self.minTCtrl.SetStyle(0, len(tMin), wx.TextAttr(colBack = bgColor))
		    #failed = True
		
		##Validate tFactor(must be a float)
		#numTfactor = None
		#try:
		    #numTfactor = float(tFactor)
		    #self.tFactCtrl.SetStyle(0, len(tFactor), wx.TextAttr(colBack = "white"))
		#except:
		    #self.tFactCtrl.SetStyle(0, len(tFactor), wx.TextAttr(colBack = bgColor))
		    #failed = True
		    
		##Validate steps(must be an int)
		#numSteps = None
		#try:
		    #numSteps = int(steps)
		    #self.kCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = "white"))
		#except:
		    #self.kCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = bgColor))
		    #failed = True
		    
		##Validate size(must be an int)
		#numSize = None
		#try:
		    #numSize = int(size)
		    #self.sizeCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = "white"))
		#except:
		    #self.sizeCtrl.SetStyle(0, len(steps), wx.TextAttr(colBack = bgColor))
		    #failed = True
		
            
        #return failed, useMC, numTmax, numTmin, numTfactor, numSteps, numSize
        

    #def OnRadioBoxChange(self, event): # wxGlade: FittingFrame.<event_handler>
	#val = self.radio_box_1.GetSelection()
	#print val
	#if val == 0:
		#self.spinPath_label.Enable(False)
		#self.pathCtrl_copy.Enable(False)
		#self.spinBrowse_btn.Enable(False)

		#self.monteCarlo_checkbox.Enable(True)
		#self.maxT_label.Enable(True)
		#self.maxTCtrl.Enable(True)
		#self.minTemp_label.Enable(True)
		#self.minTCtrl.Enable(True)
		#self.k_label.Enable(True)
		#self.kCtrl.Enable(True)
		#self.tFact_label.Enable(True)
		#self.tFactCtrl.Enable(True)

	#else:
		#print "Loading a Ground State Spin File is not yet supported!"
		#self.radio_box_1.SetSelection(0)
##		self.spinPath_label.Enable(True)
##		self.pathCtrl_copy.Enable(True)
##		self.spinBrowse_btn.Enable(True)
##
##		self.monteCarlo_checkbox.Enable(False)
##		self.maxT_label.Enable(False)
##		self.maxTCtrl.Enable(False)
##		self.minTemp_label.Enable(False)
##		self.minTCtrl.Enable(False)
##		self.k_label.Enable(False)
##		self.kCtrl.Enable(False)
##		self.tFact_label.Enable(False)
##		self.tFactCtrl.Enable(False)

        #event.Skip()

## end of class FitPanel

def ShowFittingFrame(session, procManager):
    """Creates and displays a simple frame containing the FitPanel."""
    
    frame = wx.Frame(None, -1, title = "Parameter Fitting")
    FitPanel(session, procManager, frame, -1)
    frame.Fit()
    frame.SetMinSize(frame.GetSize())
    frame.Show()
    return frame

def showFitResultFrame(data, pid):
    """Creates and displays a simple frame containing the FitResultPanel."""
    
    frame = wx.Frame(None, -1, title = "Fit Parameters")
    FitResultPanel(data, pid, frame, -1)
    frame.Fit()
    frame.SetMinSize(frame.GetSize())
    frame.Show()
    return frame

def showParamListFrame(rows, title):
    """Creates a frame with title containing a ParamListPanel displaying
    the rows data.  Returns the frame."""
    frame = wx.Frame(None, -1, title = title)
    ParamListPanel(rows, frame, -1)
    frame.Fit()
    frame.SetMinSize(frame.GetSize())
    frame.Show()
    return frame


class FitResultPanel(wx.Panel):
    def __init__(self, fitData, pid, *args, **kwds):
	self.fitSession = Session()
	self.fitSession.bondTable.data = fitData
        # begin wxGlade: FitResultPanel.__init__
        kwds["style"] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, *args, **kwds)
        self.pid_label = wx.StaticText(self, -1, " PID: " + str(pid))
        self.use_results_btn = wx.Button(self, -1, "Use Results")
        self.bond_panel = bondListGrid(self, -1, self.fitSession)

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.OnUseResults, self.use_results_btn)
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: FitResultPanel.__set_properties
        pass
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: FitResultPanel.__do_layout
        grid_sizer_1 = wx.FlexGridSizer(2, 1, 0, 0)
        grid_sizer_1_copy = wx.FlexGridSizer(1, 2, 0, 0)
        grid_sizer_1_copy.Add(self.pid_label, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1_copy.Add(self.use_results_btn, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1_copy.AddGrowableRow(0)
        grid_sizer_1_copy.AddGrowableCol(0)
        grid_sizer_1_copy.AddGrowableCol(1)
        grid_sizer_1.Add(grid_sizer_1_copy, 1, wx.EXPAND, 3)
        grid_sizer_1.Add(self.bond_panel, 1, wx.EXPAND, 0)
        self.SetSizer(grid_sizer_1)
        grid_sizer_1.Fit(self)
        grid_sizer_1.AddGrowableRow(1)
        grid_sizer_1.AddGrowableCol(0)
        # end wxGlade

    def OnUseResults(self, event): # wxGlade: FitResultPanel.<event_handler>
	#send a message to the current session object(should only effect the one in the same process)
	send(signal = "Use Fit Data", sender = "fitResultPanel", bondTable = self.fitSession.bondTable)
        event.Skip()

# end of class FitResultPanel



class ParamListPanel(wx.Panel):
    def __init__(self, rows, *args, **kwds):
        """This just shows a two column list of parameters and there values.
        Rows is a list of rows, where each row is a size two tuple or list:
        rows[n][0] = parameter n label
        rows[n][1] = parameter n value"""
        # begin wxGlade: ProcessManagerPanel.__init__
        kwds["style"] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, *args, **kwds)
        self.param_list_ctrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.rows = rows
        self.__set_properties()
        self.__do_layout()
        # end wxGlade
        

    def __set_properties(self):
        # begin wxGlade: ProcessManagerPanel.__set_properties
        pass
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: ProcessManagerPanel.__do_layout
        grid_sizer_1 = wx.FlexGridSizer(1, 1, 0, 0)
        grid_sizer_1.Add(self.param_list_ctrl, 1, wx.EXPAND, 0)
        self.SetSizer(grid_sizer_1)
        grid_sizer_1.Fit(self)
        grid_sizer_1.AddGrowableRow(1)
        grid_sizer_1.AddGrowableCol(0)
        # end wxGlade
        self.param_list_ctrl.InsertColumn(0,"Parameter", format = wx.LIST_FORMAT_CENTER, width = -1)
        self.param_list_ctrl.InsertColumn(1,"Value", format = wx.LIST_FORMAT_CENTER, width = -1)
        size = self.param_list_ctrl.GetSize()
        size[0] = self.param_list_ctrl.GetColumnCount() * self.param_list_ctrl.GetColumnWidth(0) + 12
        self.GetParent().SetMinSize(size)
        for row in self.rows:
                #preserve order of rows by insert items to end of list
                length= self.param_list_ctrl.GetItemCount()
                item = self.param_list_ctrl.InsertStringItem(length, str(row[0]))
                self.param_list_ctrl.SetStringItem(length,1, str(row[1]))


class App(wx.App):
    """Just to show the frame.  This will not actually work for fitting since fitting requires
    a valid session."""
    def __init__(self, redirect = False, filename = None):
        wx.App.__init__(self, redirect, filename)
    
    def OnInit(self):
        self.SetTopWindow(showParamListFrame([("p0", 0),("p1", 1.1)], "Parameter Snapshot, Process: 123"))
        return True


if __name__ == '__main__':       
    app = App(False)
    app.MainLoop()
    
    
#if __name__ == '__main__':
    
    
#    sess = Session()
#    sess.openXMLSession('C:\\testsess.xml')
#    stat, params, err =  fitFromFile('C:\\data.txt', sess, MCeveryTime = False)
#    print '\n\n\nans:\n', params
    
    
      #print 'start'
      #sess = Session()
      #domain = [(0,0,0,),
                #(0,0,.25*pi),
                #(0,0,.5*pi),
                #(0,0,.75 *pi),
                #(0,0, pi),
                #(0,0,1.25 *pi),
                #(0,0,1.5 *pi),
                #(0,0,1.75 * pi)]
      #sw_range = []
      #for point in domain:
          #sw_range.append(4 - 4*np.cos(point[2]))
      #sw_range = np.array(sw_range)
      #print '\n\nspinwave range:\n', sw_range
      ##sw_range = (4 - 4*np.cos(domain[))
      #sess.openXMLSession('C:\\testsess.xml')
      #print 'fitting...'
      #print fitting(sess, domain, sw_range, size = 3, k = 100)
#===============================================================================
#    def simpleFunc(p, fjac = None, y = None, err = None):
#        model = (np.cos((p**.34)/15 + 4))
#        result = (y-model)/err
#        return [0, result]
#    
#    
#    p0 = np.array([0,0,0])
#    x = np.array([1.1,2.034,3.247])
#    y = (np.cos((x**.34)/15 + 4))
#    print 'y:',y
#    err = 3*[1E-4]
#    fa = {'y':y, 'err':err}
#    
#    m = mpfit(simpleFunc, p0, functkw = fa)
# 
#    print (m.status, m.params, m.perror)
#===============================================================================

    
    
