import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
mpl.rcParams['axes.linewidth']=4

# global variables
hbarc = 197.327 #MeV fm
nmass = 0.5*(939.57+938.28) # MeV
dr = 0.0025 #grid size for gradients
npart = 4 #number of particles
ndim = 3 #number of spatial dimensions

# functions #
#Pairwise potential from notes
def pot(r):

   return (1458.*np.exp(-3.11*r) - 578.2*np.exp(-1.55*r))/r

#Total potential energy
def potential_energy(rpart):

   vsum = 0.

   for j in range(1,npart):
      for i in range(0,j):

         rpair = rpart[i][:] - rpart[j][:]
         rij = np.sqrt(np.sum(rpair**2.))
         vsum += pot(rij)

   return vsum

#Corrleation function f(r_{ij}) from notes
def cor(r,alpha):

   return (1.-np.exp(-r/alpha))*np.exp(-0.5*r/alpha)

#Generates wave function
def psi(rpart,alpha):

   amp = 1.

   rpair = np.zeros(ndim)
  
   for j in range(1,npart):
      for i in range(0,j):

         rpair = rpart[i][:] - rpart[j][:]
         rij = np.sqrt(np.sum(rpair**2.))
         fcor = cor(rij,alpha)
         amp = amp*fcor

   return amp

#Compute Laplacian of wf
def d2psi(rpart,n,alpha):

   d2 = 0.

   psi0 = psi(rpart,alpha)

   for k in range(ndim):
      rpls = np.copy(rpart)
      rmns = np.copy(rpart)

      rpls[n][k] = rpls[n][k] + dr
      rmns[n][k] = rmns[n][k] - dr
   
      d2 += (psi(rpls,alpha) + psi(rmns,alpha) - 2.*psi0)/dr**2.

   return d2

#Kinetic energy
def kinetic_energy(rpart,alpha):

   ke = 0.

   for i in range(npart):
      ke -= 0.5*hbarc**2./nmass * d2psi(rpart,i,alpha)

   return ke

#Local energy
def local_energy(rpart,alpha):

   psi0 = psi(rpart,alpha)

   ke = kinetic_energy(rpart,alpha)/psi0

   pe = potential_energy(rpart)

   return ke + pe, ke, pe

#Recenter particles at the origin
def center(rpart):

   rcm = np.zeros(ndim)

   for i in range(npart):

      rcm = rcm + rpart[i]

   rcm = rcm/npart

   return rpart - rcm

#Metropolis move
def move(rpart,step):
 
   nacc = 0

   rold = np.copy(rpart)

   psiold = psi(rold,alpha)

   rnew = np.zeros((npart,ndim))

   for i in range(npart):
      for k in range(ndim):

        drk = (np.random.uniform() - 0.5)*step
        rnew[i][k] = rold[i][k] + drk

   rnew = center(rnew)

   psinew = psi(rnew,alpha)

   ratio = (psinew/psiold)**2.

   rand = np.random.uniform()

   if (min(ratio,1) > rand):
       rold = np.copy(rnew)
       nacc = 1

   return rold,nacc

#routine to compute autocovariance
def autocov(data,tau):

   n = len(data)
   mean = np.sum(data)/n
   var = (np.sum(data**2.)/n - mean**2.)
   acov = 0.
   for i in range(0,n-tau):
      acov += (data[i] - mean)*(data[i+tau]-mean)
   acorr = 1./n * acov/var
   return acorr

#Compute averages, variance, and autocorrelation time
def stats(engs):

  nb = len(engs)
  nbm1 = max(1,len(engs)-1)
  eavg = np.sum(engs)/nb
  eavg2 = np.sum(engs**2.)/nb
  sigma = np.sqrt(1./nbm1 * (eavg2 - eavg**2.)) 
  acorrs=np.zeros(nb)
  i = 0 
  while ( i <= 5.*np.sum(acorrs)):
     acorr = autocov(engs,i)
     acorrs[i] = acorr
     i += 1
  aclen = np.sum(acorrs)
  return eavg,sigma,aclen


def printdata(i,eng,err,ke,kerr,v,verr,aclen,accept):

   print("Block =", i,"\n")
   print("Block energy = %.5f +/- %.5f" %(eng,err))
   print("Block kinetic energy = %.5f +/- %.5f" %(ke,kerr))
   print("Block potential energy = %.5f +/- %.5f" %(v,verr))
   print("Autocorrelation length = %.5f" % aclen)
   print("Acceptance =",accept*100,"%")
   print("\n"+30*"-"+"\n")

def vmc(nblk,nstep,alpha,step):

   rpart = np.zeros((npart,ndim))

   rpart = np.random.uniform(-1,1,npart*ndim).reshape(npart,ndim)

   rpart = center(rpart)

   engs = np.zeros(nblk)
   errs = np.zeros(nblk)
   acls = np.zeros(nblk)
   ts = np.zeros(nblk)
   terrs = np.zeros(nblk)
   vs = np.zeros(nblk)
   verrs = np.zeros(nblk)
   
   for i in range(nblk):

      eblk = np.zeros(nstep)
      tblk = np.zeros(nstep)
      vblk = np.zeros(nstep)
    
      nacc = 0

      if ( i == (nblk-1) ):
         confs = np.zeros((nstep,npart,ndim))
   
      for j in range(nstep):

         rpart, acc = move(rpart,step)

         nacc += acc

         eblk[j],tblk[j],vblk[j] = local_energy(rpart,alpha)

         if( i == (nblk-1) ):
            confs[j] = rpart

      engs[i], errs[i], acls[i] = stats(eblk)     
      ts[i], terrs[i], _ = stats(tblk)     
      vs[i], verrs[i], _ = stats(vblk)     
      printdata(i+1,engs[i],errs[i],ts[i],terrs[i],vs[i],verrs[i],acls[i],nacc/nstep)

   return engs,errs,ts,terrs,vs,verrs,confs

def readconfs(filename):

   rout = np.zeros((nstep,npart,ndim))

   f = open(filename,"r")

   data = f.readlines()

   for line in data:
  
      if len(line.strip().split())==1:
 
         n = int(line.strip())

         idx = data.index(line)
      
         for i in range(0,npart):
            
            rout[n][i] = [float(x) for x in data[idx+i+1].strip().split()]

   return rout

def diffuse(rconf,tau):

   rdif = np.copy(rconf)

   var = hbarc**2./nmass*dtau

   for n in range(len(rconf)):
      np.random.seed(seed=3*n+n**2) #ensure different random seed for different confs
      for i in range(npart):
         for k in range(ndim):
   
            rdif[n][i][k] = rdif[n][i][k] + np.random.normal(0.,np.sqrt(var))

   return rdif

def weight(rold,rnew,dtau,et,alpha):

    vold = potential_energy(rold)

    vnew = potential_energy(rnew)

    psiold = psi(rold,alpha)

    psinew = psi(rnew,alpha)

    return np.exp(-dtau * ( (vold+vnew)/2. - et )) * psinew/psiold

def printdatagf(tau,nconf,eng,err,ke,kerr,v,verr):

   print("tau =", tau,"\n")
   print("confs =", nconf,"\n")
   print("Energy = %.5f +/- %.5f" %(eng,err))
   print("Kinetic energy = %.5f +/- %.5f" %(ke,kerr))
   print("Potential energy = %.5f +/- %.5f" %(v,verr))
   print("\n"+30*"-"+"\n")

def gfmc(confs,dtau,ntau,nconf,et,alpha):


   engs=np.zeros(ntau+1)
   errs=np.zeros(ntau+1)
   ts=np.zeros(ntau+1)
   terrs=np.zeros(ntau+1)
   vs=np.zeros(ntau+1)
   verrs=np.zeros(ntau+1)

   evmc=np.zeros(len(confs))
   tvmc=np.zeros(len(confs))
   vvmc=np.zeros(len(confs))

   for n in range(len(confs)):
      evmc[n],tvmc[n],vvmc[n] = local_energy(confs[n],alpha)
      
   engs[0], errs[0],_ = stats(evmc)     
   ts[0], terrs[0], _ = stats(tvmc)     
   vs[0], verrs[0], _ = stats(vvmc)     

   printdatagf(0.,len(confs),engs[0],errs[0],ts[0],terrs[0],vs[0],verrs[0])

   for i in range(1,ntau+1):
  
      confsdif = diffuse(confs,dtau)
 
      #Do the branching and make 
      confsnew = np.zeros((len(confsdif)*100,npart,ndim)) #make it really big at first

      count = -1
      ng = 0

      for n in range(len(confsdif)):
         wgt= weight(confs[n],confsdif[n],dtau,et,alpha)
         rand = np.random.uniform()
         nmult = int(wgt+rand)
         ng += nmult
         if (nmult > 0):
            for j in range(0,nmult):
               count = count + 1
               confsnew[count] = confsdif[n]
     
      confsnew=confsnew[:(count+1)] #keeps only the non-zero entries 

      eg = np.zeros(ng)
      tg = np.zeros(ng)
      vg = np.zeros(ng)
            
      for n in range(ng):

         eg[n],tg[n],vg[n] = local_energy(confsnew[n],alpha)

      engs[i], errs[i],_ = stats(eg)     
      ts[i], terrs[i], _ = stats(tg)     
      vs[i], verrs[i], _ = stats(vg)     
      printdatagf(i*dtau,ng,engs[i],errs[i],ts[i],terrs[i],vs[i],verrs[i])

      et = et + 1./dtau * np.log(nconf/ng) #adjust trial energy to minimize fluctuations

      confs = np.copy(confsnew)

   return engs,errs,ts,terrs,vs,verrs

if __name__ == "__main__":

   iqmc = int(sys.argv[1]) #controls which algo (0=VMC,1=GFMC)

   #VMC parameters
   alpha = 0.9 #Correlation function parameter
   nblk = 10 #Numbr of blocks for calculation
   nstep = 1000 #Number of steps/block
   step = 1.5 #MCMC stepsize
   #GFMC parameters
   dtau = 0.002 #MeV^{-1}
   etrial = -30. #Trial energy
   ntau = 50 #Number of tau steps
   nconf = 1000 #Target number of walkers

   if (iqmc == 0):

      print("VMC calculation with the following parameters:\n")
      print("alpha = %.5f" % alpha) 
      print("Number of blocks = %d" % nblk)
      print("Number of steps/block = %d" % nstep)
      print("Step size = %.5f" % step)
      print("\n"+30*"-"+"\n")

      etots = []
      errtots = []
      engs,errs,ts,terrs,vs,verrs,confs = vmc(nblk,nstep,alpha,step)
     
      etot,errtot,_ = stats(engs)
      ktot,kerrtot,_ = stats(ts)
      vtot,verrtot,_ = stats(vs)
   
      print("\nAverage over all blocks:\n")
      print("Energy = %.5f +/- %.5f\n" % (etot,errtot))
      print("Kinetic energy = %.5f +/- %.5f\n" % (ktot,kerrtot))
      print("Potential energy = %.5f +/- %.5f\n" % (vtot,verrtot))
   
      f = open("confs.txt","w")
   
      for n in range(len(confs)):
   
         f.write("%d\n" % n)
         for i in range(npart):
               f.write("%.5f  %.5f  %.5f\n" % tuple(confs[n][i]))
   
      f.close()

      print("VMC configurations written to confs.txt")

   elif (iqmc == 1):

      print("GFMC calculation with the following parameters:\n")
      print("alpha VMC = %.5f" % alpha)
      print("Number of tau steps = %d" % ntau)
      print("Delta tau = %.5f" % dtau)
      print("Trial Energy = %.5f" % etrial)
      print("Target number of configurations = %.5f" % nconf)

      confs = readconfs("confs.txt") 
      engs,errs,ts,terrs,vs,verrs = gfmc(confs,dtau,ntau,nconf,etrial,alpha)
      taus = np.linspace(0.,dtau*ntau,ntau+1)

      fig = plt.figure(figsize=(12,8))
      plt.errorbar(taus,engs,errs,fmt="o",ms=10,capsize=4)
      plt.xlabel(r"$\tau$ (MeV$^{-1}$)",fontsize=50)
      plt.xlabel(r"$E(\tau)$ (MeV)",fontsize=50)
      plt.tick_params(axis='both',labelsize=40,direction='in',length=8,width=4)
      plt.show()
    
   else:
      print("Invalid iqmc value")
