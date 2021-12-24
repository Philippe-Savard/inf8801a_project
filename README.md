# Projet d'école Polytechnique Montréal pour le cours INF8801A
# Remis au professeur Lamia Seoud le 23 décembre 2022

# Implementation of Real-Time Facial Emotion Recognition System With Improved Preprocessing and Feature Extraction

# CONSEILS D'INSTALLATION
installer toutes les librairies du fichier "requirements.txt" à l'aide de la commande :
pip install -r PATH+requirements.txt

si la commande ne fonctionne pas, installer les librairies suivantes :

cmake
dlib
tensorflow
numpy
imutils
cv2
glob



Ensuite, il faut installer la détection des landmarks.
Télécharger le fichier via ce lien:
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

puis le placer dans le répertoire "rendu".

L'ensemble de notre travail peut se télécharger sur notre git au lien:


En particulier, on peut y trouver:
- la base de donnée FER2013 utilisée pour l'entraînement, également téléchargeable ici : https://www.kaggle.com/msambare/fer2013
- les 3 réseaux que nous avons entraîné pour ce cours, car le troisième réseau "CNN+landmarks+HOG" ne rentrait pas dans la limite de mémoire autorisée pour le rendu du projet.
- les mini rapports soumis précédemment




# CONSEILS D'UTILISATION

Une fois l'installation complétée, il suffit de lancer le main depuis un compilateur python.
Le réseau utilisé par défaut est le réseau CNN ONLY.
Les réseaux sont dans le répertoire "rendu\neuralnetwork\models".
Vous pourrez y trouver les réseaux:
-CNN ONLY
-CNN + LANDMARKS
vous pouvez ajouter le dernier réseau (trop volumineux) depuis le lien vers notre git.
Pour cela, il suffit de télécharger le dossier "cnn_landmarks_hog_model" et de le copier dans le répertoire "rendu\neuralnetwork\models".
Vous aurez ainsi la possibilité d'utiliser le troisième réseau:
-CNN + LANDMARKS + HOG

Pour sélectionner un autre réseau, il suffit de modifier une des variables ligne 23 et 24 du main.py
USE_LANDMARKS = False
USE_LANDMARKS_HOG = False
par la valeur True.
La prédiction se fera alors via un autre de nos réseaux !


# POUR CONTINUER

Il vous est possible de réentraîner nos réseaux. Ce n'est pas nécessaire, mais cela peut être utile pour améliorer les performances.
Pour cela, télécharger la base de donnée depuis notre git, l'installer dans le répertoire "dataset", et de lancer le main.
Pour changer les paramètres d'entraînement, il faut explorer le fichier "dataset/EmotionsNetwork.py" ! 

