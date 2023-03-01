#Importing necessary Libraries
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
import os
import pickle 
import operator
import librosa # main package for working with Audio Data
import librosa.display
app=Flask(__name__)

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "dihyhut9gzS66xEekElZaFhX-OEUjC3LjlqPAj3hjThN"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]



#Defining the necessary functions for creating a dataset for KNN matching.
dataset=[]
def loadDataset (filename):
    with open("F:/notebook/jeemol/project/Flask/my1.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError: 
                f.close() 
                break
loadDataset("F:/notebook/jeemol/project/Flask/my1.dat")
#Define a function to get the distance between feature vectors and find neighbors:
def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    
     #Method to calculate distance between two instances.
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance


#This function returns a list of K nearest neighbours for any instance
#to be checked within a given dataset (dataset of features.)
def getNeighbors(trainingSet , instance , k):
    distances =[]
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors  

#Identify the nearest neighbors: 
def nearestClass(neighbors): 
    classVote ={}
    for x in range(len(neighbors)): 
        response = neighbors [x]
        if response in classVote: 
            classVote [response] +=1
        else:
            classVote[response]=1
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    print("sorter is ", sorter)
    return sorter[0][0] 

@app.route('/', methods=['GET'])
def index():
# Main page
    return render_template('music.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
# Get the file from post request
        f = request.files['image']
# Save the file to ./uploads
        basepath = "F:/notebook/jeemol/project/Flask"
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename)) 
        f.save(file_path)
        print(file_path)
        i=1
        results = {1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 5: 'hiphop', 6: 'jazz', 7: 'metal', 8: 'pop', 9: 'reggae', 10: 'rock'}
        (rate, sig)=wav.read(file_path)
        print(rate, sig)
        mfcc_feat=mfcc (sig, rate, winlen=0.020, appendEnergy=False) 
        covariance = np.cov(np.matrix.transpose(mfcc_feat)) 
        mean_matrix = mfcc_feat.mean (0)
        feature=(mean_matrix, covariance, 0)
        pred=nearestClass(getNeighbors (dataset,feature, 8))
        print("predicted genre = ",pred,"class = ", results [pred]) 
        return "This song is classified as a "+str(results [pred])
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": ["image"], "values": results}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/aa6ed315-f4db-4dfd-b38f-6020ae5cc245/predictions?version=2023-02-28', json=payload_scoring,
     headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
if __name__ == '__main__':
        app.run(threaded = False ,port=8000)