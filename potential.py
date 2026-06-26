import numpy as np
from constants import *

#Minnesota potential
def minnesota(r):

   vout = np.zeros(4)

   v0r = 200. #MeV
   v0t = 178. #MeV
   v0s = 91.85 #MeV
   kr = 1.487 #fm^{-2}
   kt = 0.639 #fm^{-2}
   ks = 0.465 #fm^{-2}

   vr = v0r*np.exp(-kr*r**2)
   vt = -v0t*np.exp(-kt*r**2) 
   vs = -v0s*np.exp(-ks*r**2) 

   vout[0] = 3./8. * (vr + 0.5*vt + 0.5*vs) #central
   vout[1] = 1./8. * (-vr - 1.5*vt + 0.5*vs) #tau dot tau
   vout[2] = 1./8. * (-vr + 0.5*vt - 1.5*vs) #spin dot spin
   vout[3] = 1./8. * (-vr - 0.5*vt - 0.5*vs) #spin-isospin

   return vout

#Your potential here! 
