import tensorflow as tf
from tensorflow import keras
import zipfile
import os
import random
from shutil import copyfile
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

!wget  "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" -O "/tmp/cats-and-dogs.zip"

local_zip = "/tmp/cats-and-dogs.zip"
zip_f = zipfile.ZipFile(local_zip, "r")
zip_f.extractall("/tmp") #/tmp内に、ファイルを解凍
zip_f.close()

#data_len
print(len(os.listdir("/tmp/PetImages/Cat")))
print(len(os.listdir("/tmp/PetImages/Dog")))

#make_path
create = [
          "/tmp/cats-v-dogs",
          "/tmp/cats-v-dogs/train",
          "/tmp/cats-v-dogs/valid",
          "/tmp/cats-v-dogs/test",
          "/tmp/cats-v-dogs/train/cats",
          "/tmp/cats-v-dogs/train/dogs",
          "/tmp/cats-v-dogs/valid/cats",
          "/tmp/cats-v-dogs/valid/dogs",
          "/tmp/cats-v-dogs/test/cats",
          "/tmp/cats-v-dogs/test/dogs"
]

for dir in create:
  try:
    os.mkdir(dir)
    print(dir, "ok")
  except:
    print(dir, "no")
    
#create_dataset_function
def split_data(SOURCE, TRAIN, VALID, TEST):
  all_files = [] #全ての画像データ（画像ファイル）を格納していく
  for file_name in os.listdir(SOURCE):
    file_path = SOURCE + file_name
    if os.path.getsize(file_path): #画像のサイズ確認（容量）
      all_files.append(file_name)
    else:
      print(file_name, "no_size")
  
  train_full, test = train_test_split(all_files, test_size=0.1, shuffle=True)
  train, valid = train_test_split(train_full, test_size=0.2, shuffle=True)

  for file_name in train:
        copyfile(SOURCE + file_name, TRAIN + file_name) #ファイルの中身を学習データpathのファイルにコピー
  for file_name in valid:
        copyfile(SOURCE + file_name, VALID + file_name)
  for file_name in test:
        copyfile(SOURCE + file_name, TEST + file_name)
        
#create_dataset
SOURCE_CAT_DIR = "/tmp/PetImages/Cat/"
TRAIN_CATS_DIR = "/tmp/cats-v-dogs/train/cats/"
VALID_CATS_DIR = "/tmp/cats-v-dogs/valid/cats/"
TEST_CATS_DIR = "/tmp/cats-v-dogs/test/cats/"

SOURCE_DOG_DIR = "/tmp/PetImages/Dog/"
TRAIN_DOGS_DIR = "/tmp/cats-v-dogs/train/dogs/"
VALID_DOGS_DIR = "/tmp/cats-v-dogs/valid/dogs/"
TEST_DOGS_DIR = "/tmp/cats-v-dogs/test/dogs/"

split_data(SOURCE_CAT_DIR, TRAIN_CATS_DIR, VALID_CATS_DIR, TEST_CATS_DIR)
split_data(SOURCE_DOG_DIR, TRAIN_DOGS_DIR, VALID_DOGS_DIR, TEST_DOGS_DIR)

#Preprocessing
train_dir = "/tmp/cats-v-dogs/train" #学習データを含むディレクトリ
valid_dir = "/tmp/cats-v-dogs/valid"
test_dir = "/tmp/cats-v-dogs/test"

#学習データのみデータ拡張を行う
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
) 
valid_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)

#データを含むディレクトリを指定、画像サイズを整形、バッチにまとめる、2クラス分類
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=50,
    class_mode="binary"
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=50,
    class_mode="binary"
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=50,
    class_mode="binary"
)

#create_model
conv_base = VGG16(weights="imagenet",include_top=False,input_shape=(150, 150, 3))
model = keras.models.Sequential([
                                conv_base,
                                keras.layers.Flatten(),
                                keras.layers.Dropout(0.5),
                                keras.layers.Dense(512, activation="relu"),
                                keras.layers.Dense(1, activation="sigmoid")
])

#frozen
print(f"This is number of trainable weights before freezing the conv base:{len(model.trainable_weights)}")
conv_base.trainable = False
print(f"This is the number of trainable weights after freezing the conv base{len(model.trainable_weights)}")

#model_compile
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#model_fit
history = model.fit_generator(train_generator,epochs=15,validation_data=valid_generator) 

#fit_plot
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "b", label="training")
plt.plot(epochs, val_acc, "b", color="red", label="validation")
plt.title("train_valid_accuraccy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="training")
plt.plot(epochs, val_loss, "b", color="red", label="validation")
plt.title("train_valid_loss")
plt.legend()
plt.show()
