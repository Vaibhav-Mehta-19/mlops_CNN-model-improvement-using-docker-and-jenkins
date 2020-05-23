import pickle
import os

modelStructure = {}

 
def SaveModel():
	with open('model.data','wb') as f:
		pickle.dump(modelStructure,f)


def FC():
	modelStructure.update({'flatten' : 'Flatten'})
	modelStructure.update({'DL1' : {'Dense' : 512,'activation' : 'relu'}})
	modelStructure.update({'DL2' : {'Dense' : 1024,'activation' : 'relu'}})
	modelStructure.update({'DL3' : {'Dense' : 0,'activation' : 'softmax'}})
	modelStructure.update({'epochs' : 2})
	modelStructure.update({'learningRate' : '0.001'})
	modelStructure.update({'DenseLayers' : 3})
	SaveModel()
		
	print('\n')
	print('\n')

	for i in modelStructure:
		print(i,' : ',modelStructure[i])
		print('\n\t|\n')


def SetMobileNet():
	modelStructure.update({'model' : 'MobileNet'})
	modelStructure.update({'fineTuning' : 'n'})
	FC()


def tfLearningFunction():
	SetMobileNet()

modelStructure.update({'tf' : 'yes'})
tfLearningFunction()

os.system('git add *')
os.system("git commit -m 'added model.data'")
os.system('git push -f origin master')