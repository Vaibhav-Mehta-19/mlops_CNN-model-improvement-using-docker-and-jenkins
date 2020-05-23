import pickle
modelStructure = {}
code = []


with open('m.data','rb') as f:
	modelStructure = pickle.load(f)

for i in modelStructure:
	print(i)


lr = (modelStructure['learningRate'])
ep = (modelStructure['epochs'])


importLibs = """
import pickle
import os
from keras.applications import MobileNet
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
"""

getData = """

img_rows, img_cols = 224,224
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/dataset/train/'
validation_data_dir = '/dataset/test/'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='categorical')

"""

variables = "lr = " + str(lr) + '\nep = ' + str(ep) + '\nol = train_generator.num_classes'

model = """	
model = MobileNet(weights='imagenet',include_top = False,input_shape = (img_rows, img_cols, 3))
model.save('MobileNet.h5')
"""

fineTune = """
for l in model.layers:
	l.trainable = False
"""


makeModel = """

top_model = model.output
top_model = Flatten()(top_model)

"""

addLayers = ""

for i in range(modelStructure['DenseLayers'] - 1):
	tmp = "\ntop_model = Dense(" + str(modelStructure['DL' + str(i+1)]['Dense']) + ",activation=" + "'" + (modelStructure['DL' + str(i+1)]['activation']) + "'" + ")(top_model)"
	addLayers += tmp


tmp = "\ntop_model = Dense(ol,activation='softmax')(top_model)"

addLayers += tmp


finalModel = """

\nnmodel = Model(inputs = model.input, outputs = top_model)

"""

compileModel = """
\nnmodel.compile(loss = 'categorical_crossentropy',optimizer = keras.optimizers.Adam(learning_rate = lr), metrics = ['accuracy'])
"""



#Enter the number of training and validation samples here

trainModel = """

\nhistory = nmodel.fit_generator(
    train_generator,
    epochs = ep,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size)

"""

accuracy="""
try:
	os.system('touch result')
except:
	pass

with open('result','w') as f:
	f.write(str(history.history['accuracy'][-1]))

"""


with open('codefile.py','w') as f:
	f.write(importLibs)
	f.write(getData)
	f.write(variables)
	f.write(model)
	f.write(fineTune)
	f.write(makeModel)
	f.write(addLayers)
	f.write(finalModel)
	f.write(compileModel)
	f.write(trainModel)
	f.write(accuracy)
f.close()

import os
os.system('cat codefile.py')