from flask import Flask, render_template, request, redirect, url_for, session
import csv, socket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib
#matplotlib.use('Agg')
plt.switch_backend('agg')
from base64 import b64encode
import os

from fonctions import *

app = Flask(__name__)

# Lecture du fichier CSV et chargement dans un DataFrame
current_dir = os.path.dirname(os.path.abspath(__file__))
dataframe = pd.read_csv(os.path.join(current_dir, 'CO2_1960_2019.csv'))
# Suppression de la première ligne

#print(dataframe)

dataframe[1:]  # 'Aruba' ne contient aucune donnée

# liste vide
liste_vide=['Aruba','Samoa américaines','Bermudes','Îles Anglo-Normandes',
           'Curacao','Îles Caïmans','Îles Féroé','Gibraltar',
            'Groenland','Guam','Chine, RAS de Hong Kong','Île de Man',
            'Non classifié','Région administrative spéciale de Macao, Chine',
            'Saint-Martin (fr)','Saint-Marin','Monaco','Mariannes','Nouvelle-Calédonie',
            'Porto Rico','Cisjordanie et Gaza','Kosovo','Polynésie française',
            'Sint Maarten (Dutch part)','Îles Turques-et-Caïques','Îles Vierges (EU)',
            'Îles Vierges britanniques'
           ]
dataframe=dataframe.drop(columns=liste_vide) # supprimer les colonnes vides
#Définition de toutes les fonctions permettant de remplir le tableau de stats descriptives
liste_lineaire = ['Afrique du Nord et Moyen-Orient','Afrique du Nord et Moyen-Orient (BIRD et IDA)'
                  ,'Afrique du Nord et Moyen-Orient (hors revenu élevé)','Afrique subsaharienne'
                  ,'Afrique subsaharienne (BIRD et IDA)','Afrique subsaharienne (hors revenu élevé)','Bénin'
                  ,'Honduras','IDA totale','Indonésie','Le monde arabe','Malaisie','Maroc','Sénégal'
                  ,'Tunisie','Turquie','de dividende précoce démographique','Égypte, République arabe d’']


parametres = dataframe['Annee']





@app.route("/")
def home():
    return render_template('index.html',dataframe = dataframe)


@app.route("/page_ouverture")
def page_ouverture():
    return render_template('page_ouverture.html')

@app.route("/stats")
def statsDesc():

    nom_lien = request.args.get('nom_lien', '')
    pays = nom_lien  # Récupérer le nom du pays en supprimant les espaces et en mettant la première lettre en majuscule
    #print(pays)
    #print(dataframe.columns)
    if pays in dataframe.columns:
        donnees_pays = dataframe.loc[:, ['Annee', pays]].dropna()  # Filtrer les données pour inclure uniquement l'année et les valeurs du pays spécifié
        print(donnees_pays)
        # Reste du code pour calculer les statistiques et afficher le tableau
        valeurs_co2 = donnees_pays[nom_lien].dropna().values  # Obtenir toutes les valeurs de CO2 associées à la colonne choisie
        #print(valeurs_co2)
        # calcul des données à mettre dans le tableau
        # Définir les variables minimum et maximum
        valeurs_co2=[x for x in valeurs_co2]
        
        minimum = min(valeurs_co2)
        maximum = max(valeurs_co2)
        moyenne = Moment_r(valeurs_co2,1)
        variance = Moment_cr(valeurs_co2,2)
        skewness = round(Moment_cr(valeurs_co2,3)/(Moment_cr(valeurs_co2,2)**(1.5)),6)
        kurtosis = round((Moment_cr(valeurs_co2,4)/(Moment_cr(valeurs_co2,2)**(2))) - 3,6)

        histo_html = Histo_Continue64(valeurs_co2,15)
        nom_fichier_histogramme = str(pays) + ".png"
        graphe_html = pays_Graphe(os.path.join(current_dir, 'CO2_1960_2019.csv'),pays,True,"static/img_graph")
        nom_graphe = "graphe_"+str(pays)+".png"
        #print(nom_fichier_histogramme)
        dates = list(x for x in donnees_pays['Annee'].values)
        #print(dates)
        if pays in liste_lineaire:
            #modele_html = Droite_Regression(dates,valeurs_co2,pays,"static/modele_lin")
            #nom_modele = "modele_" + str(pays) + ".png"
            valeurs_x,theta_final = regression_polynomiale_v2(dates,valeurs_co2, 1)
              # Affichage des résultats

            def model(X, theta):
                import numpy
                return numpy.dot(model_M(X),theta)
            predicted_values = model(valeurs_x,theta_final)
            plt.figure(figsize=(10,8))
            plt.rcParams['axes.spines.right'] = False
            plt.rcParams['axes.spines.top'] = False
            plt.scatter(dates, valeurs_co2,label='observed values')
            plt.plot(dates, predicted_values*1e4, c='r',label='estimated values')

            plt.legend()
            print(theta_final[0],theta_final[1])
            plt.title(r"$\hat{a}=%.6f , \hat{b}=%.6f$"%(theta_final[0],theta_final[1]), fontsize=17)
            plt.xlabel("$x$")
            plt.ylabel("$y=ax+b$")
            plt.show()
            nom_modele = "static/modele_lin" + "/modele_" + str(pays) + ".png"
            # nom figure
            plt.savefig('modele.png')
            plt.close()
            plot_file = open('modele.png', 'rb')
            base64_string = b64encode(plot_file.read()).decode()
            plot_file.close()


            modele_html = "{}".format(base64_string)

            os.remove("modele.png")
        else:
            
            valeur_min = 1e100
            #for i in range (2,7):
            #    predicted_values, optimal_params,erreur = regression_polynomiale_v2(dates,valeurs_co2, i)
             #   if erreur < valeur_min :
              #      valeur_min = erreur
               #     order = i
            # Prédiction des valeurs estimées
            
            order = 2
            
            
            #predicted_values, optimal_params,skk = regression_polynomiale_v2(dates,valeurs_co2, order)
            valeurs_x,theta_final = regression_polynomiale_v2(dates,valeurs_co2, order)
            def model(X, theta):
                import numpy
                return numpy.dot(model_M(X),theta)
            predicted_values = model(valeurs_x,theta_final)
            
            a = "{:.3f}".format(theta_final[2])
            b = "{:.3f}".format(theta_final[1])
            c = "{:.3f}".format(theta_final[0])

            

            # Affichage des résultats
            plt.scatter(np.array(dates), np.array(valeurs_co2), color='blue', label='Données réelles')
            plt.plot(np.array(dates), np.array(predicted_values), color='red', label='Modélisation')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("Régression polynomiale degré {}; a = {}, b = {}, c = {}".format(order, a, b, c))

            plt.legend()
            nom_modele = "static/modele_lin" + "/modele_" + str(pays) + ".png"
            # nom figure
            plt.savefig('modele.png')
            plt.close()
            plot_file = open('modele.png', 'rb')
            base64_string = b64encode(plot_file.read()).decode()
            plot_file.close()


            modele_html = "{}".format(base64_string)

            os.remove("modele.png")
        """
        
        valeur_min = 100000
        for i in range (1,20):
            S,erreur = regression_polynomiale(dates,valeurs_co2,i,pays,None)
            if erreur < valeur_min :
                valeur_min = erreur
                deg = i
          
        regression_polynomiale(dates,valeurs_co2,deg,pays,"static/modele_lin")
        print(min)
        """
        nom_modele = "modele_" + str(pays) + ".png"
        return render_template("stats.html", params=dataframe, nom_colonne=pays,pays = pays,
                                donnees_pays=donnees_pays,min=minimum,max=maximum,moy=moyenne,var=variance,
                                skew=skewness,kurt=kurtosis,nom_fichier_histogramme=nom_fichier_histogramme,nom_graphe=nom_graphe
                                ,nom_modele=nom_modele,histo_html = histo_html,graphe_html = graphe_html,modele_html = modele_html)
    else:
        return redirect(url_for('bonjour'))  # Rediriger vers la page d'accueil si le nom du pays n'est pas trouvé dans les colonnes du DataFrame



if __name__ ==  '__main__':
    app.run(debug=True, threaded=False)
