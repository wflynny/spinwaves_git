from csection import plot_cross_section
import numpy as np
from numpy.random import shuffle
import matplotlib
matplotlib.use('WXAgg')
import pylab
from matplotlib._pylab_helpers import Gcf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def gaussian(x,y):
    tempout=[]
    for xval in x:
        tempin=[]
        for yval in y:
            zval = 2.0*np.exp(-xval**2/2.0)*np.exp(-yval**2/4.0)
            tempin.append(zval)
        tempout.append(tempin)
    
    return np.array(tempout).T

def everyother(arr):
    front = arr[::2]
    back = arr[1::2]
    return np.append(front,back)

def reg_mesh(lim,num,func):
    x = np.linspace(-lim,lim,num)
    y = np.linspace(-lim,lim,num)
    z = func(x,y)
    return x,y,z

if __name__ == "__main__":
    if 1:
        x = np.random.uniform(-3,3,200)
        y = np.random.uniform(-3,3,200)
        shuffle(x)
        shuffle(y)
        print x.shape
        print y.shape
        z = 2.0*np.exp(-x**2/2.0)*np.exp(-y**2/4.0)
        print z.shape
        xi = np.linspace(-3,3,100)
        yi = np.linspace(-3,3,100)
        zi = matplotlib.mlab.griddata(x,y,z,xi,yi)
        print zi.shape
        CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()
    
    # Regular mesh
    if 0:
        x_reg, y_reg, z_reg = reg_mesh(3,200,gaussian)
        plot_cross_section(x_reg,y_reg,z_reg)
        #plt.pcolormesh(z_reg)
        #plt.show()
    
    # Irregular mesh
    if 0:
        x_ireg = np.linspace(-3,3,100)
        y_ireg = np.linspace(-3,3,100)
        
###################################
        # shuffle
        if 0:
            shuffle(x_ireg)
            shuffle(y_ireg)
        #everyother
        if 1:
            x_ireg = everyother(x_ireg)
            y_ireg = everyother(y_ireg)
        
        for i in range(len(x_ireg)):
            print x_ireg[i]

##################################
        #sort data
        if 0:
            x_ireg = np.sort(x_ireg)
            y_ireg = np.sort(y_ireg)

        z_ireg = gaussian(x_ireg,y_ireg)

##################################
        #our plotter
        if 1:
            plot_cross_section(x_ireg,y_ireg,z_ireg,myFlag=True)
        #colormesh
        if 0:
            plt.pcolormesh(z_ireg)
        #contour
        if 0:
            plt.contour(x_ireg,y_ireg,z_ireg)
        plt.show()