# extract melalui notebook jika diperlukan
# !unzip flowers-recognition.zip

import os

mypath= 'flowers/'

file_name = []
tag = []
full_path = []
for path, subdirs, files in os.walk(mypath):
    for name in files:
        full_path.append(os.path.join(path, name)) 
        tag.append(path.split('/')[-1])        
        file_name.append(name)

import pandas as pd

# memasukan variabel yang sudah dikumpulkan pada looping di atas menjadi sebuah dataframe agar rapih
df = pd.DataFrame({"path":full_path,'file_name':file_name,"tag":tag})
df.groupby(['tag']).size()

tag
daisy        1538
dandelion    2110
rose         1568
sunflower    1468
tulip        1968
dtype: int64

#cek sample datanya
print(df.head())

