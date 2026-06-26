import numpy as np

#Returns the sign of a permutation
def sign(perm):

   inv = 0 #keep track of # of inversions w.r.t. a given order

   for i in range(len(perm)):
      for j in range(i+1,len(perm)):
         if (perm[i] > perm[j]):
            inv += 1
   return (-1.)**inv

#Returns index of binary in spin state
def bintoidx(state):
 
   return int("".join(map(str, state.astype(int))),2) 

#Looks for the array element that matches desired element
def find(element,array):
   for i in range(len(array)):
      if(np.all(array[i] == element)):
         return i
         break

#Take the dot product of two wave functions
def dot(psil,psir):

   dot = 0.            

   ns = len(psir)
   nt = len(psir[0])

   for i in range(ns):
      for j in range(nt):

         dot += np.conj(psil[i][j]) * psir[i][j]

   return dot

#Computes the statistics for an array of estimators
def stats(array,weights):

  nobs = len(array[0])

  tmp = np.zeros((nobs,2),dtype=complex)
  
  wtot = np.sum(weights)

  for n in range(nobs):

     obs = array[:,n]
     oavg = np.sum(weights*obs)/wtot
     oavg2 = np.sum(weights*obs**2.)/wtot
     sigma = np.sqrt(1./wtot * np.abs(oavg2 - oavg**2.)) 
 
     tmp[n] = np.array([oavg,sigma])

  return tmp

#Prints out stats
def printdata(lqmc,i,estimates,accepted):

   print("Block =", i,"\n")
   print("Block energy (MeV) = (%.5f + i %.5f) +/- %.5f " %(estimates[0][0].real,estimates[0][0].imag,estimates[0][1].real))
   print("Block kinetic energy (MeV) = (%.5f + i %.5f) +/- %.5f" %(estimates[1][0].real,estimates[1][0].imag,estimates[1][1].real))
   print("Block potential energy (MeV) = (%.5f + i %.5f) +/- %.5f" %(estimates[2][0].real,estimates[2][0].imag,estimates[2][1].real))
   print("Block potential rms p (fm) = (%.5f + i %.5f) +/- %.5f" %(estimates[3][0].real,estimates[3][0].imag,estimates[3][1].real))
   print("Block potential rms n (fm)= (%.5f + i %.5f) +/- %.5f" %(estimates[4][0].real,estimates[4][0].imag,estimates[4][1].real))
   if(lqmc==1):
      print("Acceptance =",accepted*100,"%")
   print("\n"+30*"-"+"\n")

