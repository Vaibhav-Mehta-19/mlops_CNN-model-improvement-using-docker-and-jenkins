import pickle
import os

modelStructure = {}

modelStructure.update({'tf' : 'yes'})
modelStructure.update({'model' : 'MobileNet'})
modelStructure.update({'fineTuning' : 'n'})
modelStructure.update({'flatten' : 'Flatten'})
modelStructure.update({'DL1' : {'Dense' : 512,'activation' : 'relu'}})
modelStructure.update({'DL2' : {'Dense' : 1024,'activation' : 'relu'}})
modelStructure.update({'DL3' : {'Dense' : 0,'activation' : 'softmax'}})
modelStructure.update({'epochs' : 2})
modelStructure.update({'learningRate' : '0.001'})
modelStructure.update({'DenseLayers' : 3})

with open('m.data','wb') as f:
	pickle.dump(modelStructure,f)

os.system('git add *')
os.system("git commit -m 'added file'")
os.system('git push -f origin master')