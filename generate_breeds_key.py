import os
import pandas as pd

image_path = 'img'
dog_dirs = os.listdir(image_path)
dogs_key_t = []

for i in range(len(dog_dirs)):
    index = i
    dogs = os.listdir(os.path.join(image_path, dog_dirs[i]))
    for j in range(3):
        dogs_key_t.append([index, dog_dirs[i], dogs[j]])

pd.DataFrame(dogs_key_t).to_csv('breeds.csv')