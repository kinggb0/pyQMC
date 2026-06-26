import numpy as np
import itertools #use this module to loop over permutations
from constants import *
from utils import *
from potential import *
import copy

class Walker:
      
      def __init__(self,npart,nprot):

         self.npart = npart
         self.npair = int(npart*(npart-1)/2)
         self.ns = 2**npart
         self.nprot = nprot

         self.msval = np.zeros((self.ns,self.npart))
   
         #Generate all spin states
         for i in range(self.ns):
            for j in range(self.npart):
   
               self.msval[i][j] = int(i/2**(self.npart-j-1))%2 
   

         #get isospin states
         self.tstate = np.zeros((self.ns,self.ns))
    
         self.nt = 0

         for i in range(self.ns):
         
            if(sum(self.msval[i])==self.nprot):

               self.tstate[i][self.nt] = 1
               self.nt += 1

         self.tstate = self.tstate[:,:self.nt]

         #get spin flip table
         self.nsflip = np.zeros((self.ns,self.npart),dtype=np.int64)

         for i in range(self.ns):
            for j in range(self.npart):

               flip = self.msval[i].copy()
               flip[j] = (flip[j]+1)%2
               self.nsflip[i][j] = find(flip,self.msval)  

         #make a table of pairs 
         ncount = 0 
         self.kpair = np.zeros((self.npart,self.npart),dtype=np.int64)
         self.pairtab = np.zeros((self.npair,2),dtype=np.int64)

         for j in range(1,self.npart):
            for i in range(0,j):               
           
               self.pairtab[ncount] = np.array([i,j])
               self.kpair[i][j] = ncount
               self.kpair[j][i] = ncount
               ncount += 1     

         #make a table of spin exchanges
         self.nsexch = np.zeros((self.ns,self.npair),dtype=np.int64)

         swap = np.zeros(self.npart)

         for i in range(self.ns):
            for j in range(self.npair):

               swap = self.msval[i].copy()
               n1 = self.pairtab[j][0]
               n2 = self.pairtab[j][1]
               swap[n1] = self.msval[i][n2]
               swap[n2] = self.msval[i][n1]
               self.nsexch[i][j] = find(swap,self.msval)

         #make single isospin matrix tables
         self.tauz=np.zeros((self.nt,self.nt,self.npart))
         self.tprot=np.zeros((self.nt,self.nt,self.npart))
         self.tneut=np.zeros((self.nt,self.nt,self.npart))

         tempz = np.zeros(self.ns)
         tempp = np.zeros(self.ns)
         tempn = np.zeros(self.ns)

         for n in range(self.npart):
            for jp in range(self.nt):
               for i in range(self.ns):

                  #store values of pauli z, proton, and neutron projector
                  tempz[i] = (self.msval[i][n]-0.5)*self.tstate[i][jp]
                  tempp[i] = self.msval[i][n]*self.tstate[i][jp]
                  tempn[i] = (1. - self.msval[i][n])*self.tstate[i][jp]

               for j in range(self.nt):
                  for i in range(self.ns):

                     #store them based on if the tstate is allowed
                     self.tauz[jp][j][n] += tempz[i] * self.tstate[i][j]
                     self.tprot[jp][j][n] += tempp[i] * self.tstate[i][j]
                     self.tneut[jp][j][n] += tempn[i] * self.tstate[i][j]

         #make tau dot tau table (using tau dot tau = 2P^t_{12} - 1
         self.tdt = np.zeros((self.nt,self.nt,self.npair))
         temptdt = np.zeros(self.ns)

         for k in range(self.npair):
            for jp in range(self.nt):
               for i in range(self.ns):
               
                  iex = self.nsexch[i][k]
                  temptdt[i] = 2.*self.tstate[iex][jp] - self.tstate[i][jp] #2P^t_{12} - 1 
             
               for j in range(self.nt):
                  for i in range(self.ns):
                  
                     self.tdt[jp][j][k] += temptdt[i] * self.tstate[i][j]


         #make the antisymmetrized spin-isospin state
         order = np.arange(0,self.npart,1) #make an array 1,2,...,npart

         phitemp = np.zeros((self.ns,self.ns)) #we will find which states in msval we get swapping spins, isospins

         self.phi = np.zeros((self.ns,self.nt)) #We will project onto allowed isospin states after 

         for perm in itertools.permutations(order):

            perm = np.asarray(perm) #itertools returns perm as tuple, make it an array

            sptemp = np.zeros(self.npart) #temporary spin and isospin states
            istemp = np.zeros(self.npart) 

            #Now we will antisymmetrize with respect to:
            # down_n(3) up_n(2) down_p(1) up_p(0)
    
            for i in range(self.npart):
   
               j = perm[i] 
    
               if(j==0 or j==2):
   
                  sptemp[i] = 1 #fill state i if j corresponds to up in the reference order
   
               if(j==0 or j==1):
   
                  istemp[i] = 1 #fill state i if j corresponds to proton in the reference order
   
   
            isp = bintoidx(sptemp) #convert the list of 0's and 1's to a decimal i.e. its spin state number
            itp = bintoidx(istemp)
             
            phitemp[isp][itp] = sign(perm) #Fill the state we just generated with the sign of the permutation of the order

         #Now restrict to good isospin
         for j in range(self.ns):
            for jp in range(self.nt):
               for i in range(self.ns):

                  self.phi[i][jp] += self.tstate[j][jp] * phitemp[i][j] #factor 1.j to allow complex w.f.

         self.phi = self.phi.astype(complex)


      #Generate random walker positions
      def spawn(self,nwalk):

         self.rpart = np.random.uniform(-2.,2.,3*self.npart*nwalk).reshape(nwalk,self.npart,3)

         return

      #prints out info about the spin state so we can do a sanity check
      def checkspinstate(self):

         print("Checking spin flip table:")

         for i in range(self.ns):

            print("Spin state=",self.msval[i])
            for j in range(self.npart):

               idx = self.nsflip[i][j]
               print("Particle =",j+1,"Spin flip = ",self.msval[idx])


         print("\n")

         print("Checking spin exchange table:")

         for i in range(self.ns):
            print("Sping state=",self.msval[i])
            for k in range(self.npair):
               idx = self.nsexch[i][k]
               n1 = self.pairtab[k][0]
               n2 = self.pairtab[k][1]
               print("Pair=",k+1,"n1=",n1,"n2=",n2,"Spin exchange=",self.msval[idx])

         print("\n")

         print("Number of spin state=",self.ns)
         print("Number of isospin state=",self.nt)

         print("\n")

         print("Pair table:")

         for k in range(self.npair):

            print("Pair=",k+1,"particles = ",self.pairtab[k])

         return

      #check antisymmetry of the spin state
      def atest(self):

         phi2 = dot(self.phi,self.phi)

         for k in range(self.npair):

            phitdt = np.zeros((self.ns,self.nt),dtype=complex)
            phiex = np.zeros((self.ns,self.nt),dtype=complex)
   
            for j in range(self.nt):
               for i in range(self.ns):
                  for jp in range(self.nt):
                     phitdt[i][j] += self.tdt[j][jp][k] * self.phi[i][jp]
   
               for i in range(self.ns):
                  iex = self.nsexch[i][k]
                  phiex[i][j] = 0.5*(self.phi[iex][j] + phitdt[iex][j])
            
            atest = dot(self.phi,phiex)/phi2
   
            print("Atest for pair",k,"=",atest)
         return


      #Single particle jastrow factor
      def jastrow(self,r):

         return -alpha*r**2

      #Generate the wave function:
      def setpsi(self,nw):

         x = self.rpart[nw].copy()
 
         xcm = np.average(x,axis=0) 

         x = x - xcm #center of mass

         u = 0.

         for k in range(self.npair):

            n1 = self.pairtab[k][0]
            n2 = self.pairtab[k][1]

            dx = x[n1] - x[n2]

            r = np.sqrt(np.sum(dx**2))

            u += self.jastrow(r)

         self.psi = np.exp(u)*self.phi

         return          

      #Do a VMC step with the walker
      def vmcstep(self):
     
         nwalkers = len(self.rpart)

         self.nacc = 0    

         for nw in range(nwalkers):

            self.setpsi(nw)
           
            psi = self.psi
   
            rsave = self.rpart[nw].copy()     
        
            dr = np.zeros((self.npart,3))
        
            #move wp's coordinates
            for n in range(self.npart):
               for k in range(3):
        
                  dr[n][k] = (np.random.uniform() - 0.5)*step
        
            self.rpart[nw] += dr
        
            self.setpsi(nw) 
       
            psip = self.psi #Get the new psi
    
            rat = dot(psip,psip).real/dot(psi,psi).real #drop the 0j
        
            rand = np.random.uniform()
   
            if (min(rat,1)>rand):
        
               self.nacc += 1
        
            else: 
              
               self.rpart[nw] = rsave

         self.weights = np.ones(nwalkers)
        
         return  

      #Compute propagator
      def prop(self,nw,dtau):

         self.setpsi(nw)

         tmp = np.zeros((self.ns,self.nt),dtype=complex)
      
         wtc = 0. #Weight for central potential

         for k in range(self.npair):

            n1 = self.pairtab[k][0]
            n2 = self.pairtab[k][1]
            dx = self.rpart[nw][n1] - self.rpart[nw][n2]
            r = np.sqrt(np.sum(dx**2))
            
            v = minnesota(r)   
   
            wtc += v[0] #central
   
#            psitdt = np.zeros((self.ns,self.nt),dtype=complex)
#            psisds = np.zeros((self.ns,self.nt),dtype=complex)
#            psisdstdt = np.zeros((self.ns,self.nt),dtype=complex)
#   
#            for j in range(self.nt):
#               for i in range(self.ns):
#                  for jp in range(self.nt):
#                     psitdt[i][j] += self.tdt[j][jp][k] * self.psi[i][jp]
#      
#               for i in range(self.ns):
#                  iex = self.nsexch[i][k]
#                  psisdstdt[i][j] = 2.*psitdt[iex][j] - psitdt[i][j]
#                  psisds[i][j] = 2.*self.psi[iex][j] - self.psi[iex][j]
#  
#            #do e^{-Ht} ~ 1 - Ht 
#            tmp += -v[1]*psitdt*dtau #tau tau
#            tmp += -v[2]*psisds*dtau #spin spin
#            tmp += -v[3]*psisdstdt*dtau #spin-isospin

         return self.psi, wtc

      #Do a GFMC propagation
      def gfmcstep(self,dtau,etrial):

         #Diffusion step
         sig = np.sqrt(hbarc**2/mnuc * dtau)

         nconfs = len(self.rpart)
         
         self.weights = np.zeros(nconfs)

         for nw in range(nconfs):

            wts = np.zeros(2)

            rsave = self.rpart[nw].copy()
            
            dr = np.zeros((self.npart,3))

            self.setpsi(nw)

            psi2 = dot(self.psi,self.psi).real

            for n in range(self.npart):
               for k in range(3):
                  dr[n][k] = np.random.normal(0.,sig)
 
            #Shift by +dr
            self.rpart[nw] += dr 

            psip, wtc = self.prop(nw,dtau)

            wts[0] = np.exp(-wtc*dtau).real * np.sqrt(dot(psip,psip).real/psi2)

            #Shift by +dr
            self.rpart[nw] = rsave

            self.rpart[nw] -= dr 

            psip, wtc = self.prop(nw,dtau)

            wts[1] = np.exp(-wtc*dtau).real * np.sqrt(dot(psip,psip).real/psi2)

            self.rpart[nw] = rsave

            #Check for negative weights
            for i in range(2):
               if(wts[i]<0.):
                  wts[i]=0.

            #Heat bath sampling
            wttot = np.sum(wts) 
            wts = wts/wttot
            rand = np.random.uniform()
            if(rand<wts[0]):
               self.rpart[nw] += dr
            else:
               self.rpart[nw] -= dr

            self.weights[nw] = wttot*0.5*np.exp(etrial*dtau)
            
         return

      def branch(self):

         temp =[]

         for nw in range(len(self.weights)):

            if (self.weights[nw] > 0.):
               mult = int(self.weights[nw] + np.random.uniform())
               for i in range(mult):

                  temp.append(self.rpart[nw])

         self.rpart = np.asarray(temp)
     
         return

      def laplace(self,nw,n):

         self.setpsi(nw)

         psi0 = self.psi

         d2psi = 0.j

         rsave = self.rpart.copy()

         for k in range(3):

            #Move the particle by +grid spacing
            self.rpart[nw][n][k] += h

            self.setpsi(nw)

            psip = self.psi

            self.rpart = rsave.copy()
           
            #Move the particle by -grid spacing
            self.rpart[nw][n][k] -= h

            self.setpsi(nw)

            psim = self.psi

            self.rpart = rsave.copy()

            #Finite difference for the derivative
            d2psi += dot(psim + psip - 2.*psi0,psi0)/h**2

         return d2psi
         
      #calculate operators
      def calcops(self):
      
         nwalkers = len(self.rpart)

         out = np.zeros((nwalkers,5),dtype=complex)

         for nw in range(nwalkers):

            self.setpsi(nw)
   
            psi = self.psi
   
            psi2 = dot(psi,psi)
   
            #Local Kinetic energy
            ke = 0.j
            for n in range(self.npart):
              
               ke += -0.5*hbarc**2/mnuc * self.laplace(nw,n)/psi2
   
            out[nw][1] = ke
         
            #Potential energy
            tmp = np.zeros((self.ns,self.nt),dtype=complex)
   
            for k in range(self.npair):
              
               n1 = self.pairtab[k][0]
               n2 = self.pairtab[k][1]
               dx = self.rpart[nw][n1] - self.rpart[nw][n2]
               r = np.sqrt(np.sum(dx**2))
              
               v = minnesota(r)   
   
               tmp += v[0] * self.psi #central
   
               psitdt = np.zeros((self.ns,self.nt),dtype=complex)
               psisds = np.zeros((self.ns,self.nt),dtype=complex)
               psisdstdt = np.zeros((self.ns,self.nt),dtype=complex)
   
               for j in range(self.nt):
                  for i in range(self.ns):
                     for jp in range(self.nt):
                        psitdt[i][j] += self.tdt[j][jp][k] * self.psi[i][jp]
      
                  for i in range(self.ns):
                     iex = self.nsexch[i][k]
                     psisdstdt[i][j] = 2.*psitdt[iex][j] - psitdt[i][j]
                     psisds[i][j] = 2.*self.psi[iex][j] - self.psi[iex][j]
   
               tmp += v[1]*psitdt #tau tau
               tmp += v[2]*psisds #spin spin
               tmp += v[3]*psisdstdt #spin-isospin
           
            out[nw][2] = dot(psi,tmp)/psi2
    
            #Total energy
            out[nw][0] = out[nw][1] + out[nw][2] 

            #Radii
            psip = np.zeros((self.ns,self.nt),dtype=complex)
            psin = np.zeros((self.ns,self.nt),dtype=complex)
            for n in range(self.npart):

               x = self.rpart[nw][n].copy()
 
               xcm = np.average(x,axis=0) 

               x = x - xcm      

               for j in range(self.nt):
                  for i in range(self.ns): 
                    for jp in range(self.nt):
                        psip[i][j] += self.tprot[j][jp][n] * np.sum(x**2.) * psi[i][jp] 
                        psin[i][j] += self.tneut[j][jp][n] * np.sum(x**2.) * psi[i][jp] 

            out[nw][3] = np.sqrt( dot(psi,psip) )
            out[nw][4] = np.sqrt( dot(psi,psin) )

         return out

      def copy(self):
         return copy.deepcopy(self)
