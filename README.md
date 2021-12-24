# Projet de session du cours d'Applications multimédias - INF8801A
# Implémentation de l'article : Real-Time Facial Emotion Recognition System With Improved Preprocessing and Feature Extraction
https://ieeexplore.ieee.org/abstract/document/9214207?casa_token=Nle3kbDqfgcAAAAA:GJ8M-hG8MgQA9vOT3D8akbq-sUG4-fwZZ3J8IW0VyhmRkNWg-ZGl2j-q2vwnwOwdpuSrp1cNfh8 

# Installation

Installez toutes les librairies du fichier "requirements.txt" à l'aide de la commande :
pip install -r PATH+requirements.txt
Si la commande ne fonctionne pas, installez les librairies suivantes :
1.	cmake
2.	dlib
3.	tensorflow
4.	numpy
5.	imutils
6.	cv2
7.	glob

Ensuite, il faut télécharger le fichier de prédiction des points clés du visage. Ce fichier se nomme shape_predictor_68_face_landmarks.dat. 

Vous pouvez télécharger le fichier via le lien suivant (le fichier est également disponible sur le git du travail):

https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat 

Ensuite, placez ce fichier dans le répertoire du projet.

# Entraînement

Si vous souhaitez entraîner les réseaux, il faut télécharger le dataset « FER2013 » et insérer son contenu dans le dossier « dataset/FER2013 » prévu à cet effet dans le répertoire du projet. Vous pouvez télécharger le dataset directement sur le git du projet ou au lien suivant : https://www.kaggle.com/msambare/fer2013 
*note : le fichier FER2013 doit contenir uniquement les fichiers « train » et « test ». Ceux-ci contiennent les sous-fichiers correspondant aux 7 émotions.

Alternativement, les 3 réseaux que nous avons entraînés dans le cadre de ce projet sont disponibles directement sur le git du projet. Veuillez noter que le troisième réseau "CNN+landmarks+HOG" dépassait la limite de mémoire autorisée pour le rendu du projet et n’est donc pas inclus dans le fichier de remise. 

# Conseils d'utilisation

Une fois l'installation complétée, il suffit de lancer le fichier main depuis un compilateur python.
Le réseau utilisé par défaut est le réseau CNN ONLY.
Les réseaux déjà entraînés fournis dans le fichier de remise sont dans le répertoire "remise\neuralnetwork\models".

Vous pourrez y trouver les réseaux:
-	CNN ONLY
-	CNN + LANDMARKS

Comme il a été mentionné ci-haut, vous pouvez ajouter le troisième réseau (trop volumineux) depuis notre git.
Pour cela, il suffit de télécharger le dossier "cnn_landmarks_hog_model" et de le copier dans le répertoire "remise\neuralnetwork\models".

Vous aurez ainsi la possibilité d'utiliser le troisième réseau:
-	CNN + LANDMARKS + HOG

Pour sélectionner un autre réseau que celui par défaut, il suffit de modifier une des variables des ligne 20 et 21 du main.py (USE_LANDMARKS = False ou USE_LANDMARKS_HOG = False) par la valeur True.
La prédiction se fera alors via un autre de nos réseaux.

# Pour aller un peu plus loin

Il vous est possible de réentraîner nos réseaux. Ce n'est pas nécessaire, mais cela peut être utile pour améliorer les performances.
Pour cela, téléchargez la base de données depuis notre git comme il a été mentionné précédemment. Assurez-vous de l'installer dans le répertoire "dataset". Dans le EmotionsNetwork.py, ajoutez le paramètre « True » à la fonction self.load_data() à la ligne 135. Cela permet d’assurer que vous extrayez de nouveau les données du dataset. 

Vous pouvez ensuite lancer le main.py. Cela va prendre un certain temps (quelques minutes). Des logs interactifs sont imprimés dans la console pendant l’exécution.
Vous pouvez retirer le paramètre « True » de la fonction self.load_data() une fois les données extraites (pour sauver du temps).
Pour changer les paramètres d'entraînement, il faut explorer le fichier "dataset/EmotionsNetwork.py" et y modifier les différents paramètres. 

