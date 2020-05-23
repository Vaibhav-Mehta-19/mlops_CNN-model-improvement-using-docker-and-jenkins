import os
import pickle
os.system('sudo docker cp mydocker:/data/result .')


maxAccuracy = -1
accuracy = None


with open('result','r') as f:
	accuracy = float(f.read())
	print('accuracy after 0 : ',accuracy)


if accuracy > maxAccuracy:
	maxAccuracy = accuracy
	os.system('sudo docker cp mydocker:/data/history .')
	os.system('sudo docker cp mydocker:/data/classifier.h5 .')


modelStructure = []

var=0

while(accuracy < 0.8 and var < 5):


	if accuracy > maxAccuracy:

		maxAccuracy = accuracy
		os.system('sudo docker cp mydocker:/data/history .')
		os.system('sudo docker cp mydocker:/data/classifier.h5 .')

	os.system('sudo docker cp mydocker:/data/result .')
	
	with open('result','r') as f:
		accuracy = float(f.read())
		print('accuracy after ' + str(var) + ' : ',accuracy)
	
	with open('m.data','rb') as f:
		modelStructure = pickle.load(f)

	var += 1
	
	if var == 1:
		dlCount = int(modelStructure['DenseLayers'])
		modelStructure.update({'DL' + str(dlCount) : {'Dense' : 128, 'activation' : 'relu'}})
		dlCount = int(modelStructure['DenseLayers']) + 1
		modelStructure.update({'DenseLayers' : dlCount})
	
	elif var == 2:
		dlCount = int(modelStructure['DenseLayers'])
		modelStructure.update({'DL' + str(dlCount) : {'Dense' : 256, 'activation' : 'relu'}})
		dlCount = int(modelStructure['DenseLayers']) + 1
		modelStructure.update({'DenseLayers' : dlCount})
	
	elif var == 3:
		dlCount = int(modelStructure['DenseLayers'])
		modelStructure.update({'DL' + str(dlCount) : {'Dense' : 512, 'activation' : 'relu'}})
		dlCount = int(modelStructure['DenseLayers']) + 1
		modelStructure.update({'DenseLayers' : dlCount})

	elif var == 4:
		dlCount = int(modelStructure['DenseLayers'])
		modelStructure.update({'DL' + str(dlCount) : {'Dense' : 1024, 'activation' : 'relu'}})
		dlCount = int(modelStructure['DenseLayers']) + 1
		modelStructure.update({'DenseLayers' : dlCount})

	with open('m.data','wb') as f:
		pickle.dump(modelStructure,f)

	os.system("python3 create.py")
	os.system('sudo docker exec mydocker python3 /data/codefile.py')

print('max accuracy is ',maxAccuracy)