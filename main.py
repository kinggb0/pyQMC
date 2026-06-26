import numpy as np
from constants import *
from walker import *
from utils import *

np.random.seed(505)

nav = 10
neq = 10
nstep = 10
nwalk = 100
nest = 5 #Number of observables to compute estimators for
lqmc = 2 #Save this for adding features
etrial = 0. #Trial energy in MeV

#initialize a walker
w = Walker(4,2)  #initiailze a walker with A particles, Z protons, and n confs

#Sanity check that our spin operators are working
w.checkspinstate()

print("\n")

#Tests the antisymmetry of the spin state
w.atest()

print("\n")

w.spawn(nwalk) #for now, we just spawn one random walker. Could make parallel/read in configs in the future

#zero estimators here
est = np.zeros((nav,nest,2),dtype=complex)

#Do the Metropolis
for i in range(neq+nav):

   #Alerts us that we did the equlibration
   if(i==neq):

      print("Equilibration done!")

   nacc = 0

   tmp = np.zeros((nstep,nest),dtype=complex)

   nw=0
   
   #Do the MC stepping
   for j in range(nstep):

      if(lqmc==1):
         w.vmcstep()
         nacc += w.nacc
         if(i>=neq):
            tmp[j] = stats(np.array(w.calcops()),w.weights)[:,0]

      if(lqmc==2):
         w.gfmcstep(dtau,etrial)
         if(i>=neq):
            tmp[j] = stats(np.array(w.calcops()),w.weights)[:,0]
         w.branch()
         ng = len(w.rpart)
         etrial = etrial + 1./dtau * np.log(nwalk/ng)
                     

   if(i>=neq):

      est[i-neq] = stats(tmp,np.ones(nstep)) # block average
 
      printdata(lqmc,i-neq+1,est[i-neq],nacc/(nwalk*nstep)) 

      #Add a way to save configs here


