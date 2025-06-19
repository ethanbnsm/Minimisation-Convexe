"""
Methodes de gradients.
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import matplotlib.pyplot as plt
import itertools


################################
##### FUNCTION DEFINITIONS #####
################################
def function(A,b,c,xx):
    shape=xx[0].shape
    ZZ=np.zeros_like(xx[0])
    for item in itertools.product(*map(range,shape)):
        vect=list()
        for elem in xx:
            vect.append(elem[item])
        vec=np.array(vect,ndmin=1)
        ZZ[item]=0.5*np.dot(vec.T,A.dot(vec))-np.dot(vec.T,b) + c
    return ZZ

def gradient(A,b,x):
    return A.dot(x)-b

def f(x, A, b):
    return 0.5*(np.transpose(x).dot(A)).dot(x)-np.transpose(x).dot(b)

def scalaireA(x,y):
    return np.vdot(A.dot(x),y)

def pas_fixe(x0,fonction,pas=1.0e-2,tol=1.0e-10,itermax=10000):
    A=fonction['A']
    b=fonction['b']
    c=fonction['c']


    #***** Initialisation *****
   
    k=0
    xx = [x0]
    dir = -1 * gradient(A,b,xx[k])
    residu = [np.linalg.norm(-1 * dir)]

   
    while residu[k] >= tol and k <= itermax:
        #----- Calcul de x(k+1) -----
        xx.append(xx[k] + pas * dir)

        #----- Calcul de la nouvelle direction de descente d(x+1) -----
        dir = -gradient(A,b,xx[-1])

        #----- Calcul du residu r(k+1) -----
        residu.append(np.linalg.norm(gradient(A,b,xx[-1])))
        k += 1

    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

def conjugate(x0,fonction,tol=1.0e-10,itermax=10000):
    A=fonction['A']
    b=fonction['b']
    c=fonction['c']

    #***** Initialisation *****
    xx = [x0]
    dir = -1 * gradient(A,b,xx[0])
    residu = [np.linalg.norm(-1 * dir)]

    k=0
    while residu[k] >= tol and k <= itermax:
        #----- Calcul de rho(k) -----
        rho = - np.vdot((gradient(A,b,xx[k])),dir)/scalaireA(dir,dir)
       
        #----- Calcul de x(k+1) -----
        xx.append(xx[k] + rho * dir)
       
        #----- Calcul de la nouvelle direction de descente d(x+1) -----
        beta = scalaireA(gradient(A,b,xx[-1]),dir)/scalaireA(dir,dir)
        dir = -gradient(A,b,xx[-1])+beta*dir
       
        #----- Calcul du residu r(k+1) -----
        residu.append(np.linalg.norm(-gradient(A,b,xx[-1])))
        k += 1

    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

def plot(xx,res):
    plt.figure()
    ax1=plt.gca()
   
    X1, X2 = np.meshgrid(np.linspace(-5.0,5.0,101),np.linspace(-5.0,5.0,101))
    Z=function(A,b,c,[X1,X2])
   
    ax1.contour(X1,X2,Z)

    ax1.plot(xx.T[0,0],xx.T[0,1],'k-x')

    ax1.set_aspect('equal')
   
    plt.figure()
    plt.plot(res)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel(r'$||\nabla f(x) ||_2$')
    plt.title('Convergence')

    plt.show()

#######################
##### SCRIPT PART #####
#######################

###############################
##### SELF-SUSTAINED PART #####
###############################
if __name__=="__main__":
    #***** Exercice 01 *****
    """
    Pour cet exercice le minimum est [-0.1429,-0.4286]
    """
    #x0=np.array([[8],[4]])
    #A=np.array([[4,1],[1,2]])
    #b=np.array([[-1],[-1]])
    #c=np.zeros(1)

    #fonction={'A':A, 'b':b, 'c':c}

    #----- Pas fixe -----
    """cas_01=pas_fixe(x0,fonction,1.0e-1,1.0e-6,10000)
    print(cas_01['xx'][-1], len(cas_01['xx'])-1)

    plot(cas_01['xx'],cas_01['residu'])"""

    #----- Gradient conjugue -----
    #cas_02=conjugate(x0,fonction,1.0e-6,10000)
    #print(cas_02['xx'][-1], len(cas_02['xx'])-1)

    #plot(cas_02['xx'],cas_02['residu'])

    #***** Exercice 02 *****
   
    x0=np.array([[800],[400]])
    A=np.array([[16,0],[0,16]])
    b=np.array([[1634],[-1914]])
    c=np.zeros(1)

    fonction={'A':A, 'b':b, 'c':c}
   
    cas_02=conjugate(x0,fonction,1.0e-6,10000)
    print(cas_02['xx'][-1], len(cas_02['xx'])-1)

    plot(cas_02['xx'],cas_02['residu'])
    #***** Exercice 03 *****
    #----- Import data -----
    """
    Les donnees sont stockees dans la liste data sous forme de dictionnaire ayant les cles b'A' et
    b'b'
    """
    #data=list()
    #for ind in range(1,6):
     #   data.append(pickle.load(open('paire_%i.pickle' % ind,'rb'), encoding='bytes'))
