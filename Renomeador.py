import os
import shutil
import numpy as np
import pandas as pd
import string
from PIL import Image

obv = np.asarray(pd.read_csv('OBV32.csv',header=None))

url = 'archive/train'
lista1 = os.listdir(url)

lista1 = sorted(lista1, key=str.lower)
cont = 0

newUrl = 'all/archive/train/'
targets = []
for i in range(len(lista1)):
    lista = os.listdir(url+'/'+lista1[i])
    # for j in range(len(lista)):
    for j in range(50):
        image_path = url+'/'+lista1[i]+'/'+lista[j]
        image = Image.open(image_path)
        resized_image = image.resize((50, 50))
        resized_image.save(newUrl+str(cont)+'.jpg')
        targets.append(lista1[i])
        cont = cont + 1
        
# Lista de letras do alfabeto
letras = list(string.ascii_lowercase)

# Dicionário para mapear cada letra para seu vetor de posições
vetores = {}
for i, letra in enumerate(letras):
    vetores[letra] = obv[i]

finalTargets = []
for i in range(len(targets)):
    finalTargets.append(vetores[targets[i]])

matrizIndices = np.asarray(finalTargets)
# matrizIndices = np.transpose(matrizIndices)
dfIndices = pd.DataFrame(data=matrizIndices)
        
dfIndices.to_csv('all/archive/train/targets/targetsTrain.csv', index=False)      