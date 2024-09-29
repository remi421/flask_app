import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib
#matplotlib.use('Agg')
plt.switch_backend('agg')
from base64 import b64encode
import os



# Moments

def Moment_r(data,r):
    import functools
    data=[x for x in data] # transformation de données en liste
    fonc_r= lambda x : x**r
    S=functools.reduce( lambda x, y: x+y,map(fonc_r,data))
    return round(S/(1.0*len(data)),4)

#Moments Centrés

def Moment_cr(data,r):
    data=[x for x in data] # transformation de données en liste
    import functools
    m=Moment_r(data,1)
    fonc_r= lambda x : (x-m)**r
    S=functools.reduce(lambda x, y: x+y,map(fonc_r,data))
    return round(S/(1.0*len(data)),4)

#Creer un histogrammex

def Histo_Discrete(data,nom=None,chemin_image=None):
    
    import numpy
    import matplotlib.pyplot as plt
    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
    
    # sous fonction pour compter les occurrences
    def comptage(data):
        data=sorted(data)
        Dic_compt={}
        for valeur in data:
            Dic_compt[valeur]=data.count(valeur)
        return Dic_compt

    D=comptage(data)
    
    valeurs=[k for k in D.keys()]
    effectifs=[v for v in D.values()]
    i_mode=numpy.argmax(effectifs)
    ### multi_mode
    indice_mode=[i for i in range(len(effectifs)) if effectifs[i]==effectifs[i_mode]]

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    
    ax1.set_xticks(valeurs)  ## positions des extrémités des classes
    ax1.set_xticklabels(valeurs ,fontsize=10, rotation=25)
    ax1.set_xlabel("Valeurs",fontsize=14)
    ax1.set_ylabel("Effectifes",fontsize=14)
    
    for k in range(len(valeurs)):
        if k not in indice_mode:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white")
        else:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white",
                    color = [0.15,0.15,0.85],hatch="X", lw=1., zorder = 0)
        for i in range(len(valeurs)):
            ax1.text(valeurs[i], effectifs[i], "%d"%(effectifs[i]),fontsize=9,
                     horizontalalignment='center',verticalalignment='bottom',style='italic')
    
    if nom is None:
        plt.show()
    else:
        nom_fig=str(nom)+".png"
        if nom is not None and chemin_image is not None:
            nom_fichier_histogramme = chemin_image + "/" + str(nom) + ".png"
            plt.savefig(nom_fichier_histogramme, format="png")
        plt.close()



#####################################################################
### histogramme Histogramme des données Discrètes en base64 pour flask
#####################################################################
### histogramme Histogramme des données Discrètes en base64 pour flask

def Histo_Discrete64(data):
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('Agg')
    plt.switch_backend('agg')
    from base64 import b64encode
    import os

    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
   
    # sous fonction pour compter les occurrences
    def comptage(data):
        data=sorted(data)
        Dic_compt={}
        for valeur in data:
            Dic_compt[valeur]=data.count(valeur)
        return Dic_compt

    D=comptage(data)
   
    valeurs=[k for k in D.keys()]
    effectifs=[v for v in D.values()]
    i_mode=numpy.argmax(effectifs)
    ### multi_mode
    indice_mode=[i for i in range(len(effectifs)) if effectifs[i]==effectifs[i_mode]]

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)

    #ax1 =plt.gca()
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
   
    ax1.set_xticks(valeurs)  ## positions des extrémités des classes
    ax1.set_xticklabels(valeurs ,fontsize=10, rotation=25)
    ax1.set_xlabel("Valeurs",fontsize=14)
    ax1.set_ylabel("Effectifes",fontsize=14)
   
    for k in range(len(valeurs)):
        if k not in indice_mode:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white",color = numpy.random.rand(3))
        else:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white",
                    color = [0.15,0.15,0.85],hatch="X", lw=1., zorder = 0)
        for i in range(len(valeurs)):
            ax1.text(valeurs[i], effectifs[i], "%d"%(effectifs[i]),fontsize=9,
                     horizontalalignment='center',verticalalignment='bottom',style='italic')
   
    # nom figure
    plt.savefig('histo64.png')
    plt.close()
    plot_file = open('histo64.png', 'rb')
    base64_string = b64encode(plot_file.read()).decode()
    plot_file.close()


    img_elem = "{}".format(base64_string)

    os.remove("histo64.png")

    return img_elem

#####################################################"

### histogramme Histogramme des données Continues en base64 pour flask

def Histo_Continue64(data,k):
    # k=nombre d'intervalles
   
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('Agg')
    plt.switch_backend('agg')
    from base64 import b64encode
    import os
   
    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
   
    data=numpy.array([x for x in data])
    Ext=[min(data)+(max(data)-min(data))*i/(1.0*k) for i in range(k+1)]
    C=[0.5*(Ext[i]+Ext[i+1]) for i in range(k)]

    NN=[] # Effectifs des classes
    for i in range(k):
        NN.append(((Ext[i] <= data) & (data<=Ext[i+1])).sum())
       
    # pour la classe modale
    indice_max=[i for i in range(k) if NN[i]==numpy.max(NN)]
   
    TT=[str("{:.4f}".format(t)) for t in Ext]  ## pour afficher les extrémités des classes
   
    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(111)
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    largeur=Ext[1]-Ext[0]  #  largeur des classes
   
    for i in range(k):
       
        if i in indice_max:
            ax1.bar(C[i], NN[i],largeur,  color = [0.15,0.15,0.80], edgecolor="white", hatch="/",
                    lw=1., zorder = 0,alpha=0.9)
        else:
            ax1.bar(C[i], NN[i],largeur, align='center', color = numpy.random.rand(3),edgecolor="white")
       
        ax1.text(C[i], NN[i], "%d"%(NN[i]),fontsize=9, style='italic',
                 horizontalalignment='center',verticalalignment='bottom')

    ax1.set_xticks(Ext)  ## positions des extrémités des classes
    ax1.set_xticklabels(TT ,fontsize=9, rotation=45)
    ax1.set_xlim(numpy.min(data)-0.75*largeur, numpy.max(data)+0.75*largeur)
    ax1.set_ylim(0.0, numpy.max(NN)+3.0)
    ax1.set_xlabel("Valeurs",fontsize=13,labelpad=0)
    ax1.set_ylabel("Effectifs",fontsize=14)

    # nom figure
    plt.savefig('histo64.png')
    plt.close()
    plot_file = open('histo64.png', 'rb')
    base64_string = b64encode(plot_file.read()).decode()
    plot_file.close()


    img_elem = "{}".format(base64_string)

    os.remove("histo64.png")

    return img_elem



#Creer un graphe 
def pays_Graphe(fichier,pays, figure=True,chemin = None):
    
    import pandas
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('Agg')
    plt.switch_backend('agg')
    from base64 import b64encode
    import os
    #lecture
    
    data_CO2=pandas.read_csv(fichier, sep=',')
    

    # 
    data=data_CO2[[data_CO2.columns[0],pays]][data_CO2[pays].notnull()]
    
    fig= plt.subplots(1,1, figsize=(10,7))
    ax=plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    plt.plot([i for i in range(len(data))],data[data.columns[1]])
    
    plt.xticks([i for i in range(len(data[data.columns[0]]))],data[data.columns[0]].values,
               rotation='vertical')
    
    # pour le titre
    annee_d=data[data.columns[0]].values[0]
    annee_f=data[data.columns[0]].values[len(data)-1]
    
    plt.title("{}. Emission de CO2 entre {} et {}".format(pays,annee_d,annee_f),fontsize=15)
    
    plt.xlabel("Date",labelpad=10,fontsize=13)
    
    # nom figure
    plt.savefig('graphe.png')
    plt.close()
    plot_file = open('graphe.png', 'rb')
    base64_string = b64encode(plot_file.read()).decode()
    plot_file.close()


    img_elem = "{}".format(base64_string)

    os.remove("graphe.png")

    return img_elem

#faire une droite de régression

def Droite_Regression(X,Y,nom,chemin = None):
    import numpy
    import matplotlib.pyplot as plt
    n=len(X)
    # vecteur (1,1,...,1)
    Un=numpy.ones(n)
    # transformer X en array (format numpy)
    XX=numpy.array([t for t in X])
    # transformer Y en array (format numpy)
    YY=numpy.array([t for t in Y])
    # Matrice M
    M=numpy.vstack((Un,XX)).T
    # produit: transpose(M)*M
    S1=numpy.matmul(M.T, M)
    # inverse de : transpose(M)*M
    S2=numpy.linalg.inv(S1)
    # produit: transpose(Y)*M
    S3=numpy.matmul(YY.T, M)
    # solutions (Estimateurs)
    S=numpy.matmul(S2, S3)
    
    # graphique
    
    plt.plot(XX, YY,'o', label='données brutes', markersize=10)
    plt.plot(XX,S[1]*XX + S[0], 'r', label="droite d'ajustement")
    plt.xlabel("X", fontsize=15)
    plt.ylabel("Y=aX+b", fontsize=15)
    plt.title("$\hat{a}=%.4f , \hat{b}=%.4f$"%(S[1],S[0]), fontsize=16)
    plt.legend()
    plt.savefig('modele.png')
    plt.close()
    plot_file = open('modele.png', 'rb')
    base64_string = b64encode(plot_file.read()).decode()
    plot_file.close()


    img_elem = "{}".format(base64_string)

    os.remove("modele.png")

    return img_elem

def regression_polynomiale(X_data, Y_data, degree,nom,chemin = None):
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Création de la matrice des caractéristiques polynomiales
    X_poly = np.column_stack([X_data ** i for i in range(degree + 1)])

    # Calcul des coefficients de la régression polynomiale
    S1 = np.matmul(X_poly.T, X_poly)
    S2 = np.linalg.inv(S1)
    S3 = np.matmul(Y_data.T, X_poly)
    S = np.matmul(S2, S3)  # Estimateur

    # Prédiction des valeurs y à partir des données d'entrée
    y_pred = np.matmul(X_poly, S)

    # Calcul de l'erreur entre le modèle et les données d'origine
    erreur = np.sqrt(np.mean((Y_data - y_pred) ** 2))  # RMSE

    # Affichage des résultats
    plt.scatter(X_data, Y_data, color='blue', label='Données réelles')
    plt.plot(X_data, y_pred, color='red', label='Régression polynomiale')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Régression polynomiale degré =  : {}".format(degree))
    plt.legend()

    if chemin is not None:
        nom_modele = chemin + "/modele_" + str(nom) + ".png"
        plt.savefig(nom_modele, format="png")
    plt.close()
    
    return list(S),erreur
#params : nombre de parametres du polynome a estimer
def loss_function(params, X):

    lambda_reg = 10000
    order = len(params) - 1
    T = len(X)
    loss = 0
    for t in range(order, T):
        if order == 1:
            predicted_value = (params[0] + params[1] * (t-1)) * X[t-1]
        elif order == 2:
            predicted_value = (params[0] + params[1] * (t-1) + params[2] * (t-1)**2) * X[t-1]
        elif order == 3:
            predicted_value = (params[0] + params[1] * (t-1) + params[2] * (t-1)**2 + params[3] * (t-1)**3) * X[t-1]
        loss += (X[t] - predicted_value)**2

    # Terme de régularisation L2
    regularization_term = lambda_reg * np.sum(np.square(params))
    
    # Ajout du terme de régularisation à la perte
    loss += regularization_term
    
    return loss

def get_estimated_values(X, optimal_params):
    order = len(optimal_params) - 1
    T = len(X)
    estimated_values = [X[0]]
    for t in range(1, T):
        if order == 1:
            predicted_value = (optimal_params[0] + optimal_params[1] * (t-1)) * X[t-1]
        elif order == 2:
            predicted_value = (optimal_params[0] + optimal_params[1] * (t-1) + optimal_params[2] * (t-1)**2) * X[t-1]
        elif order == 3:
            predicted_value = (optimal_params[0] + optimal_params[1] * (t-1) + optimal_params[2] * (t-1)**2 + optimal_params[3] * (t-1)**3) * X[t-1]
        estimated_values.append(predicted_value)
    return estimated_values



def regression_polynomiale_v(X, order):
    num_iterations = 1000
    learning_rate=0.00001
    # valeur initiale des paramètres
    initial_params = np.zeros(order+1)
    optimal_params = initial_params.copy()
    # Optimiser loss function
    result = minimize(loss_function, initial_params, args=(X,), method='BFGS')

    # Retrieve the optimal coefficients
    optimal_params = result.x
    #optimal_params = gradient_descent(initial_params,X,learning_rate,num_iterations,100)

    print(optimal_params)
    valeurs_estimees = get_estimated_values(X, optimal_params)
    print(len(valeurs_estimees))
    return valeurs_estimees,optimal_params



# Pour gradient 
def model_M(X):
    import numpy
    n=len(X)
    return numpy.vstack((numpy.ones(n),X)).T

def Fonction_objectif(X, Y, theta):
    import numpy
    import math
    #a=theta[0]
    #b=theta[1]
    #print(X)
    S=sum([(t2-(sum([theta[p]*(t1)**p for p in range(len(theta))]) ))**2  for t1,t2 in zip(X,Y)])/(2.0*len(X))
    #S=sum([numpy.exp(2*numpy.log(numpy.abs(t2-a-b*t1)))  for t1,t2 in zip(X,Y)])
    #X = numpy.array(X)
    #Y = numpy.array(Y)
    #S=numpy.sum(numpy.power(Y-a-b*X,2))
    print(S)
    return S

def grad_M(X, Y, theta):
    import numpy
    n = len(theta) - 1  # Degré du polynôme (le dernier coefficient est le terme constant)

    derivatives = []
    for i in range(n + 1):
        if i == 0:
            derivative_i = -2.0 * sum([t2 - sum([theta[j] * (t1 ** j) for j in range(n + 1)]) for t1, t2 in zip(X, Y)])
        else:
            derivative_i = -2.0 * sum([t1 ** i * (t2 - sum([theta[j] * (t1 ** j) for j in range(n + 1)])) for t1, t2 in zip(X, Y)])

        derivatives.append(derivative_i)

    return numpy.array(derivatives)

    return numpy.array(derive)

def gradient_descent(X, Y, theta, pas_gradient,tolerance=1e-6):
    import numpy
    
    n_iterations=0
    historique_couts= [] # Cout du modele
    
    while True:
        n_iterations +=1
        theta = theta - pas_gradient * grad_M(X, Y, theta) # theta=theta- alpha* grad(psi)
        
        historique_couts.append(Fonction_objectif(X, Y, theta)/(2.0*len(X))) #  cout
        
        #print(n_iterations,theta, grad_M(X,Y,theta))
        
        if numpy.max(numpy.abs(grad_M(X,Y,theta))) < tolerance:
            theta_f=theta
            break
        
    return theta_f, historique_couts

def model(X, theta):
    import numpy
    return numpy.dot(model_M(X),theta)

def regression_polynomiale_v2(dates,valeurs_co2, order):
    valeurs_x = [t+1 for t in range(len(valeurs_co2))]
    
    co2 = [1e-4*x for x in valeurs_co2]
    
    import numpy
    theta = numpy.random.rand(order+1)
    pas_gradient = 1e-5
    theta_final, historique_couts = gradient_descent(valeurs_x, co2, theta, pas_gradient,tolerance=1e-3)
    #y_chap = model(valeurs_x, theta_final)
    
    return valeurs_x,theta_final