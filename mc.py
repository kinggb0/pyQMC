import numpy as np
from walker import *
from constants import *

def vmcstep(w):

    nacc = 0

    for j in range(nstep):

       wp = w.copy() #make a copy of the walker

       dr = np.zeros((w.npart,3))

       #move wp's coordinates
       for n in range(w.npart):
          for k in range(3):

             dr[n][k] = (np.random.uniform() - 0.5)*step

       wp.rpart += dr

       wp.setpsi()
       w.setpsi()

       rat = dot(wp.psi,wp.psi).real/dot(w.psi,w.psi).real #drop the 0j


       rand = np.random.uniform()

       if (min(rat,1)>rand):

          nacc += 1

          print("accepted")

       return wp.copy()
