import numpy as np

def grad_lag1(xx,mu):
    """
    Renvoie le gradient du lagrangien du probleme.

    ENTREE :
      * xx : un tableau numpy (n,1)
      * mu : un tableau numpy (m,1)

    SORTIE :
      * un tableau numpy (n,1)
    """
    x=xx[0,0]
    y=xx[1,0]
    z=xx[2,0]

    #m=mu[0,0]
    m1=mu[0,0]
    m2=mu[1,0]

    #grad_L = np.array([[4*x**3+2*m*x],
    #                   [4*y**3+2*m*y]])

    grad_L = np.array([[-z+m2*z],
                       [-z+2*m1*y],
                       [-(x+y)+2*m1*z+m2*x]])
    return grad_L

def hess_lag1(xx,mu):
    """
    Renvoie la hessienne du lagrangien du probleme.

    ENTREE :
      * xx : un tableau numpy (n,1)
      * mu : un tableau numpy (m,1)

    SORTIE :
      * un tableau numpy (n,n)
    """
    x=xx[0,0]
    y=xx[1,0]
    z=xx[2,0]

    #m=mu[0,0]
    m1=mu[0,0]
    m2=mu[1,0]

    #hess_L = np.array([[12*x**2+2*m,0],
    #                   [0,12*y**2+2*m]])
    
    hess_L= np.array([[0,0,m2-1],
                     [0,2*m1,-1],
                     [m2-1,-1,2*m1]])
    return hess_L

def vecteur1(xx,mu):
    """
    Renvoie le vecteur a droite du probleme lineaire.

    ENTREE :
      * xx : un tableau numpy (n,1)

    SORTIE :
      * un tableau numpy (n+f,1) avec f le nombre de contraintes egalites
      ici f=1
    """
    x=xx[0,0]
    y=xx[1,0]
    z=xx[2,0]

    #m=mu[0,0]
    m1=mu[0,0]
    m2=mu[1,0]

    #vect1 = -np.array([[4*x**3+2*m*x],
    #                   [4*y**3+2*m*y],
    #                   [x**2+y**2-1]])
    
    vect1= -np.array([[-z+m2*z],
                      [-z+2*m1*y],
                      [-(x+y)+2*m1*z+m2*x],
                      [z**2+y**2-1],
                      [x*z-3]])
    return vect1

def matrice1(xx,mu):
    """
    Renvoie la matrice a gauche du probleme lineaire.

    ENTREE :
      * xx : un tableau numpy (n,1)

    SORTIE :
      * un tableau numpy (n+f,n+f) avec f le nombre de contraintes egalites
    """
    x=xx[0,0]
    y=xx[1,0]
    z=xx[2,0]

    #m=mu[0,0]
    m1=mu[0,0]
    m2=mu[1,0]

    #mat1 = np.array([[12*x**2+2*m,0,2*x],
    #                 [0,12*y**2+2*m,2*y],
    #                 [2*x,2*y,0]])

    mat1=np.array([[0,0,m2-1,0,z],
                   [0,2*m1,-1,2*y,0],
                   [m2-1,-1,2*m1,2*z,x],
                   [0,2*y,2*z,0,0],
                   [z,0,x,0,0]])
    return mat1

#***** Initialisation *****
#----- Liste pour les x(k) -----
xx=[np.array([[0.1],[0.1],[0.1]])]

#----- Liste pour les multiplicateurs de lagrange -----
mu=[np.array([[0.1],[0.1]])]

#----- Liste pour le residu -----
residu=[np.linalg.norm(grad_lag1(xx[-1],mu[-1]))]

#----- Parametres de l'algorithme -----
tol=1.0e-10
itermax=10000
k=0
#***** Boucle *****
while (residu[-1] >= tol) and (k<itermax) :
    #----- Resolution du probleme lineaire -----
    delta=np.dot(np.linalg.inv(matrice1(xx[-1],mu[-1])),vecteur1(xx[-1],mu[-1]))

    #----- Avancement des points -----
    delta_xx=np.array([delta[0],delta[1],delta[2]])
    delta_mu=np.array([delta[3],delta[4]])
    xx.append(xx[-1]+delta_xx)
    mu.append(mu[-1]+delta_mu)

    #----- Residu -----
    residu.append(np.linalg.norm(grad_lag1(xx[-1],mu[-1])))
    k+=1

#***** Resultats *****
print("Résultats :")
print("k=")
print(k)
print("x=")
print(xx[-1])
print("valeurs et vecteurs propres de la matrice hessienne :")
print(np.linalg.eig(hess_lag1(xx[-1],mu[-1])))
