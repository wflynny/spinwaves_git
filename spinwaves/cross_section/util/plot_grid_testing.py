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
        h_list = np.linspace(0.1,6.3,25)
        k_list = np.zeros(h_list.shape)
        l_list = np.zeros(h_list.shape)  
        kapvect=np.empty((len(h_list),3),'Float64')
        kapvect[:,0]=h_list
        kapvect[:,1]=k_list
        kapvect[:,2]=l_list
        w_list = np.linspace(-10,10,25)
              
        x = np.array([np.sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]) for k in kapvect])
        y = np.array(w_list[:11])
        z = np.array([[5.12860269706866e-17, 6.54425595165001e-16, 1.91261465949042e-15, 3.77366949371352e-15, 6.16176772039836e-15, 8.98209286255106e-15, 1.21264437527016e-14, 1.54793336936620e-14, 1.89240888975080e-14, 2.23484914068263e-14, 2.56495909613761e-14, 2.87374206679589e-14, 3.15374775019772e-14, 3.39919546849215e-14, 3.60598245636361e-14, 3.77159569027891e-14, 3.89495121246938e-14, 3.97618704165391e-14, 4.01643485587899e-14, 4.01759229487253e-14, 3.98211277524156e-14, 3.91282401625953e-14, 3.81278084359380e-14, 3.68515291654967e-14, 3.53314423661443e-14],[7.38315878305832e-17, 9.42113937498472e-16, 2.75340839521337e-15, 5.43259104137565e-15, 8.87050765114381e-15, 1.29306600112138e-14, 1.74572812495681e-14, 2.22841162138712e-14, 2.72432008107923e-14, 3.21729855800144e-14, 3.69252628784401e-14, 4.13705159747222e-14, 4.54014899901125e-14, 4.89349660344607e-14, 5.19118805195658e-14, 5.42960558491763e-14, 5.60718873200365e-14, 5.72413618556358e-14, 5.78207711424278e-14, 5.78374336845972e-14, 5.73266689744905e-14, 5.63291849819445e-14, 5.48889591103359e-14, 5.30516219130951e-14, 5.08632983352005e-14],[1.15303536052406e-16, 1.47130884692734e-15, 4.30002568674238e-15, 8.48413227187518e-15, 1.38531613474678e-14, 2.01939422758400e-14, 2.72632123604795e-14, 3.48013292516212e-14, 4.25459817290961e-14, 5.02448763698686e-14, 5.76665558015424e-14, 6.46087524372881e-14, 7.09039652502155e-14, 7.64222304594741e-14, 8.10713078631171e-14, 8.47946985438621e-14, 8.75680324791038e-14, 8.93944126673134e-14, 9.02992819996921e-14, 9.03253040600081e-14, 8.95276376560128e-14, 8.79698568351515e-14, 8.57206415522338e-14, 8.28512535031970e-14, 7.94337264802500e-14],[2.04760421861816e-16, 2.61280642814749e-15, 7.63614979885639e-15, 1.50664460310309e-14, 2.46009728646835e-14, 3.58611737421027e-14, 4.84150534785228e-14, 6.18015292755212e-14, 7.55547788527081e-14, 8.92267699163451e-14, 1.02406471626985e-13, 1.14734689480918e-13, 1.25913968759041e-13, 1.35713514817018e-13, 1.43969523982301e-13, 1.50581663320327e-13, 1.55506655614455e-13, 1.58750011287848e-13, 1.60356912798167e-13, 1.60403123766455e-13, 1.58986597309601e-13, 1.56220230648453e-13, 1.52225988269113e-13, 1.47130419412153e-13, 1.41061444436184e-13],[4.59279280533790e-16, 5.86054592768653e-15, 1.71279456927170e-14, 3.37941601722371e-14, 5.51801808913451e-14, 8.04369024326650e-14, 1.08595356106597e-13, 1.38621329471106e-13, 1.69469979387811e-13, 2.00136365802121e-13, 2.29698543220360e-13, 2.57350835468711e-13, 2.82426049209028e-13, 3.04406510189435e-13, 3.22924805449006e-13, 3.37755887404910e-13, 3.48802684910572e-13, 3.56077557889670e-13, 3.59681850959773e-13, 3.59785502534990e-13, 3.56608222247884e-13, 3.50403239476936e-13, 3.41444121549845e-13, 3.30014719435630e-13, 3.16401959532148e-13],[1.80679795116707e-15, 2.30553016947724e-14, 6.73810870573883e-14, 1.32945730296487e-13, 2.17078022034080e-13, 3.16437594016114e-13, 4.27212537634217e-13, 5.45334276315176e-13, 6.66692412482332e-13, 7.87333526705200e-13, 9.03630698938255e-13, 1.01241441093442e-12, 1.11105993389035e-12, 1.19753074489439e-12, 1.27038144674885e-12, 1.32872670556474e-12, 1.37218464487563e-12, 1.40080388843994e-12, 1.41498312449626e-12, 1.41539088827241e-12, 1.40289151423935e-12, 1.37848120305669e-12, 1.34323616457789e-12, 1.29827306435045e-12, 1.24472066661381e-12],[8.21089291935825e-14, 1.04773537803160e-12, 3.06209607034812e-12, 6.04164486042966e-12, 9.86499012198132e-12, 1.43803306753109e-11, 1.94144364512704e-11, 2.47824132476315e-11, 3.02974663298966e-11, 3.57799347482175e-11, 4.10649951359215e-11, 4.60086104969741e-11, 5.04915014889751e-11, 5.44211194595158e-11, 5.77317791358826e-11, 6.03832470113030e-11, 6.23581689218975e-11, 6.36587545473558e-11, 6.43031220532054e-11, 6.43216526514950e-11, 6.37536255421096e-11, 6.26443125106248e-11, 6.10426213159829e-11, 5.89992982036701e-11, 5.65656392375065e-11],[1.80679795466384e-15, 2.30553017393922e-14, 6.73810871877937e-14, 1.32945730553782e-13, 2.17078022454200e-13, 3.16437594628529e-13, 4.27212538461019e-13, 5.45334277370583e-13, 6.66692413772609e-13, 7.87333528228958e-13, 9.03630700687088e-13, 1.01241441289378e-12, 1.11105993604063e-12, 1.19753074721202e-12, 1.27038144920747e-12, 1.32872670813628e-12, 1.37218464753128e-12, 1.40080389115097e-12, 1.41498312723473e-12, 1.41539089101167e-12, 1.40289151695442e-12, 1.37848120572452e-12, 1.34323616717751e-12, 1.29827306686305e-12, 1.24472066902277e-12],[4.59279280985679e-16, 5.86054593345277e-15, 1.71279457095694e-14, 3.37941602054875e-14, 5.51801809456374e-14, 8.04369025118075e-14, 1.08595356213445e-13, 1.38621329607496e-13, 1.69469979554554e-13, 2.00136365999037e-13, 2.29698543446363e-13, 2.57350835721921e-13, 2.82426049486909e-13, 3.04406510488944e-13, 3.22924805766735e-13, 3.37755887737231e-13, 3.48802685253762e-13, 3.56077558240018e-13, 3.59681851313667e-13, 3.59785502888986e-13, 3.56608222598754e-13, 3.50403239821701e-13, 3.41444121885795e-13, 3.30014719760334e-13, 3.16401959843459e-13],[2.04760421996545e-16, 2.61280642986667e-15, 7.63614980388086e-15, 1.50664460409444e-14, 2.46009728808705e-14, 3.58611737656988e-14, 4.84150535103792e-14, 6.18015293161856e-14, 7.55547789024219e-14, 8.92267699750549e-14, 1.02406471694366e-13, 1.14734689556412e-13, 1.25913968841890e-13, 1.35713514906315e-13, 1.43969524077030e-13, 1.50581663419408e-13, 1.55506655716776e-13, 1.58750011392303e-13, 1.60356912903679e-13, 1.60403123871998e-13, 1.58986597414211e-13, 1.56220230751244e-13, 1.52225988369275e-13, 1.47130419508962e-13, 1.41061444529000e-13],[1.15303536109370e-16, 1.47130884765420e-15, 4.30002568886671e-15, 8.48413227606657e-15, 1.38531613543116e-14, 2.01939422858163e-14, 2.72632123739482e-14, 3.48013292688140e-14, 4.25459817501150e-14, 5.02448763946909e-14, 5.76665558300312e-14, 6.46087524692065e-14, 7.09039652852440e-14, 7.64222304972287e-14, 8.10713079031685e-14, 8.47946985857529e-14, 8.75680325223647e-14, 8.93944127114767e-14, 9.02992820443024e-14, 9.03253041046312e-14, 8.95276377002418e-14, 8.7969856786110e-14, 8.57206415945821e-14, 8.28512535441277e-14, 7.94337265194923e-14]])

        #xi = x
        #yi = y
        #zi = matplotlib.mlab.griddata(x,y,z,xi,yi)
        CS = plt.contourf(x,y,z)
        plt.colorbar()
        plt.show()
    if 0:
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