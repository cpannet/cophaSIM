import numpy as np


def ABCDmod():  #r=|coef refl en chp|, t=|coef trans en chp|
    
    i = 1j
    A2P_ABCD = np.array([[1,i],[1,1],[1,-i],[1,-1]])/2
    #V2PM_ABCD=np.array([[1,i,-i,1],[1,1,1,1],[1,-i,i,1],[1,-1,-1,1]])/4
    
    return A2P_ABCD


def realisticABCDmod(phaseshifts,transmissions):
    """
    Creates the A2P matrix with phaseshifts not equal to integer numbers of Pi/2.
    A,B,C and D are given in (float) number of Pi/2.
    Thus for the perfect CHIP, (kA,kB,kC,kD)=(-1,0,1,2)
    """
    kA,kB,kC,kD = phaseshifts
    t1,t2,t3,t4 = np.sqrt(transmissions)        # Convert in amplitude

    i = 1j
    A2P_ABCD = np.array([[1,np.exp(kA*np.pi/2*i)],
                         [1,np.exp(kB*np.pi/2*i)],
                         [1,np.exp(kC*np.pi/2*i)],
                         [1,np.exp(kD*np.pi/2*i)]])/2
    
    A2P_ABCD = np.dot(np.diag([t1,t2,t3,t4]), A2P_ABCD)
    # V2PM_ABCD=np.array([[1,i,-i,1],[1,1,1,1],[1,-i,i,1],[1,-1,-1,1]])/4
    
    return A2P_ABCD

def ABCDmodphot(R):
    r=np.sqrt(R)
    t=np.sqrt(1-r**2)
    rp=-r
    A2P = np.array([[r*t,t*r*1j],[t*r,r*t],[r*rp,t*t*1j],[t*t,r*rp]])
    return A2P





#""" MAIN of Test """
#
if __name__ == '__main__':

    ich = np.reshape([[0,1]], [2,1])
    V2PM, P2VM, ich = coh_fs_abcdphot(1,0.64)
    # print(np.abs(np.dot(P2VM, V2PM)))   # Verify the matrix is truely inversible

