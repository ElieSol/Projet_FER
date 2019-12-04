# Projet_FER

## Test technique de Machine Learning - Reconnaissance des émotions

### 1. Téléchargements des jeux de données

Par soucis de droits de diffusion, les jeux de données n’ont pas été inclus dans ce dossier. Ils sont disponible dans les liens suivants:

Dataset FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Dataset JAFFE: http://www.kasrl.org/jaffe.html

Dataset FEI: 
https://fei.edu.br/~cet/frontalimages_manuallyaligned_part1.zip
https://fei.edu.br/~cet/frontalimages_manuallyaligned_part2.zip


### 2. Lancement et test des modèles

..* Classification binaire

Pour lancer le script de classification binaire (binary_classification.py, mettre en argument le chemin du fichier/image test:

```python
python binary_classification.py path/to/image_to_test
```
..* Classification en classes multiples

* Conversion des images contenues dans le csv en images au format JPEG
```python
python csv_to_images.py
```

* Entraînement du modèle
```python
python train_emotion_classifier.py
```
* Test du modèle avec une image
```python
python facial_emotion_image.py path/to/image_to_test
```
Source du code original (qui a été utilisé dans le dossier MultiClass_Classification/Kaggle_Code):
https://github.com/abhijeet3922/FaceEmotion_ID

