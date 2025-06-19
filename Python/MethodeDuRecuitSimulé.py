"""TP sur la méthode de recuit simulé.

Le problème est celui dit du "voyageur de commerce" pour lequel un voyageur de commerce doit se rendre dans une série de villes et cherche à minimiser la distance totale parcourue.

Dans le TP proposé ici, on impose au voyageur de partir de Paris et d'y revenir.

Pour n villes à visiter, il y a donc n! parcours possibles. La fonction coût est donc définie comme la
distance totale parcourue par le voyageur de commerce. Attention de bien prendre en compte le départ et le retour de et vers Paris.

Le travail demandé consiste à trouver l'ordre dans lequel le voyageur de commerce doit parcourir les villes
pour minimiser le trajet qu'il a à effectuer. Pour ce faire, vous utiliserez deux méthodes :
  * une méthode "brute force" pour laquelle vous calculerez tous les trajets possibles et sélectionnerez le
    plus court
  * la méthode du recuit simulé

Pour la méthode "brute force", on pourra s'aider des fonctions suivantes :
  * itertools.permutations() (print itertools.permutations.__doc__ pour l'aide sur la fonction)
  * les méthodes argmin() et min() des tableaux numpy
  * numpy.random.random() (print numpy.random.random.__doc__ pour l'aide sur la fonction)

Le dictionnaire "dico" possède les coordonnées des différentes villes impliquées.

On s'attachera à comparer les temps de calcul des deux méthodes en utilisant time.time.

"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import copy
import sys

################################
##### FUNCTION DEFINITIONS #####
################################
def cost_function(city_list, city_dict):
    """
    Fonction coût à minimiser. Pour une liste de villes donnée en entrée, la fonction calcule la distance à
    parcourir pour rallier toutes ces villes une à une. Attention, la fonction prend en compte la première
    distance de Paris et la dernière distance vers Paris.

    Entrée :
      * city_list : liste ordonnée des villes à parcourir, liste python
      * city_dict : dictionnaire des villes contenant les coordonnées permettant de
                    calculer les distances à parcourir, dictionnaire python

    Sortie :
      * la distance parcourue, float python

    """

    return ???
    
def compute_new_candidate(list_in):
    """
    Fonction associée à la méthode de recuit simulé permettant de calculer un nouveau trajet
    candidat. Afin de respecter les théorèmes de convergence, la matrice de transition afférente
    doit posséder certaines propriétés comme l'irréductibilité ou l'apériodicité. La manière de
    trouver un nouveau candidat proposée dans le corps de la fonction permet de respecter ces critères.

    Entrée :
      * list_in : une liste représentant le parcours actuel du voyageur de commerce, liste python

    Sortie :
      * une liste représentant un parcours candidat, liste python
    """
    #----- Matrice de transition Q1 -----
    # On tire au hasard le nouveau trajet par une permutation de la liste actuelle.
    # Voir la fonction numpy.random.permutation et la méthode tolist()

    # OU
    
    #----- Matrice de transition Q3 -----
    # On tire au hasard deux villes dans la listes, d'indices i et j, et :
    # * si i<j, la nouvelle chaîne est construite en inversant le sens
    #           de parcours entre i et j, inclus. Par exemple :
    #           chaîne d'entrée = [0,1,2,3,4,5,6,7,8,9]
    #           i=3, j=7
    #           chaîne de sortie = [0,1,2,7,6,5,4,3,8,9]
    # * si i>=j, la nouvelle chaîne est construite en inversant le sens
    #           de parcours à l'extérieur de  i et j, inclus. Par exemple :
    #           chaîne d'entrée = [0,1,2,3,4,5,6,7,8,9]
    #           i=7, j=3
    #           chaîne de sortie = [9,8,7,4,5,6,3,2,1,0]
        
    return ???
    
def compute_Temp(h,k,ind,Temp):
    """
    Fonction associée à la méthode de recuit simulé. Permet de calculer la nouvelles valeur de
    température à la fin d'une itération (voir algorithme du cours).

    Entrée :
      * h>0 : paramètre du calcul, float python. Plus h est petit, plus l'algorithme risque de rester
              piéger dans un minimum local. Plus h est grand, plus longue est la convergence de
              l'algorithme
      * k : paramètre de l'algorithme, integer python
      * ind : itération courante de l'algorithme, integer python
      * Temp : température courante de l'algorithme

    Sortie : 
      * nouvelle valeur du paramètre k de l'algorithme, integer python
      * nouvelle valeur de température
    """

    return ???

def plot_chemin(chemin,dico):
    fig=plt.figure(figsize=(5,5))
    
    chemin=['Paris']+chemin+['Paris']
    plt.plot([dico[ville]['x'] for ville in chemin],[dico[ville]['y'] for ville in chemin],'bo-')
    plt.plot(dico['Paris']['x'],dico['Paris']['y'],'ro')

    plt.xlim([-500.0,500.0])
    plt.ylim([-800.0,300.0])
    plt.title("Trajet")
    ax=plt.gca()
    ax.set_aspect('equal')

def plot_temp(Temp_list):
    plt.figure()
    plt.plot(Temp_list)
    plt.xlabel('$n$')
    plt.ylabel('$T$')
    plt.title(u'Profil de température')
    plt.grid()

##################
##### SCRIPT #####
##################

##### Paramètres #####
#***** Dictionnaire des villes *****
dico=dict()

dico['Lille']      ={'x':52.0, 'y':197.0}
dico['Orléans']    ={'x':-33.0, 'y':-105.0}
dico['Lyon']       ={'x':185.0, 'y':-343.0}
dico['Paris']      ={'x':0.0, 'y':0.0}
dico['Marseille']  ={'x':225.0, 'y':-617.0}
dico['Strasbourg'] ={'x':403.0, 'y':-31.0}
dico['Rennes']     ={'x':-300.0, 'y':-88.0}
dico['Metz']       ={'x':285.0, 'y':30.0}

dico['Bordeaux']   ={'x':-213.0, 'y':-448.0}
dico['Perpignan']  ={'x':40.0, 'y':-688.0}
dico['Cherbourg']  ={'x':-289.0, 'y':86.0}

######################
##### QUESTION 1 #####
######################
#***** Liste non ordonnée des villes à parcourir *****
parcours=np.random.permutation(['Marseille',
                                'Lyon',
                                'Rennes',
                                'Lille',
                                'Orléans',
                                'Strasbourg',
                                'Metz']).tolist()

###### Résolution du problème en force brute #####
print("##### Résolution du problème en force brute #####")

#***** Calcul de toutes les permutations possibles *****
t1=time.time()
permutations=list()
???

print('Nombre de trajets étudiés : ', len(permutations))
#***** Calcul de la fonction coût pour chaque permutation *****
cost=list()
???

t2=time.time()
cost=np.asarray(cost)
print('Trajet le plus court :')
for ind, item in zip(range(1,len(parcours)+1),permutations[cost.argmin()]):
    print(ind,' ',item)
print('Distance totale : ', cost.min())
print('Temps de calcul : ',t2-t1)

##### Résolution du problème par la méthode du recuit simulé #####
print("##### Résolution du problème par la méthode du recuit simulé #####")

#***** Paramètres du calcul *****
#----- Initialisation -----
candidate=parcours

#----- Paramètres de l'algorithme -----
itermax=10000
hpar=3.0
kpar=1
Temp=1.0/kpar
Temp_list=[Temp]

#***** Algorithme de résolution *****
t1=time.time()
for ind in range(itermax):
    #----- Calcul d'un nouveau trajet candidat -----
    new_candidate=compute_new_candidate(candidate)

    #----- Calcul de la différence de coût entre l'ancien et le nouveau trajet -----
    delta=cost_function(new_candidate,dico)-cost_function(candidate,dico)

    #----- Si le nouveau trajet candidat est plut cher, il peut quand même -----
    #----- être accepté avec une certaine probabilité -----
    ???
    
    #----- Diminution de la température -----
    kpar, Temp = compute_Temp(hpar,kpar,ind+2,Temp)
    Temp_list.append(Temp)

t2=time.time()

#***** Résultat *****
print('Trajet le plus court :')
for ind, item in zip(range(1,len(parcours)+1),candidate):
    print(ind,' ',item)
print('Distance totale : ', cost_function(candidate,dico))
print('Temps de calcul : ',t2-t1)

#----- Evolution des chemins -----
plot_chemin(parcours,dico)
plot_chemin(candidate,dico)

#----- Profil de température -----
plot_temp(Temp_list)

plt.show()

######################
##### QUESTION 2 #####
######################
#***** Liste non ordonnée des villes à parcourir *****
parcours=np.random.permutation(['Marseille',
                                'Lyon',
                                'Rennes',
                                'Lille',
                                'Orléans',
                                'Strasbourg',
                                'Metz',
                                'Bordeaux',
                                'Perpignan',
                                'Cherbourg']).tolist()

