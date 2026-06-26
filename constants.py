### VMC parameters ###

alpha = 0.1 #Exponential of simple correlation function
step = 1. #MC step size
h = 0.0025 #Grid spacing for derivatives 

### GFC Parameters ###

dtau = 0.002 #time step for a GFMC calculation in MeV^{-1}

### Physical constants ###

hbarc = 197.327 #hbar times c in MeV fm
mpi0 = 139.98 #nuetral pion mass
mpic = 139.57 #charged pion mass
mpi = (2.*mpic + mpi0)/3. #averge pion mass
mprot = 938.27 #proton mass
mneut = 939.57 #neutron mass
mnuc = 0.5*(mprot + mneut) #average nucleon mass

