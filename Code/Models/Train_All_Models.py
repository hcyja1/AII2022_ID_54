import numpy as np
import pickle
from random import *
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from numpy import load
from xgboost import XGBClassifier
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

#Function to load features that were previously extracted and saved
def loadFeatures(cnn_model,Database):
    data = load(r"../Feature Extraction/"+Database+"/Features/"+cnn_model+"_"+Database+"_features.npy")
    return data

#Function to load previously saved CNN model
def loadModel(modelName):
    loaded_model = load_model(r"../Feature Extraction/Models/" + modelName + "_Model.h5")
    return loaded_model

#Function to load previously saved labels
def loadLabels(Database):
    labels = load(r"../Feature Extraction/"+Database+"/"+Database+"_labels.npy")
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_encoded = le.transform(labels)
    return labels_encoded

#Driver function
def train_all_models(Database):
    #Encode and load labels
    labels_encoded = loadLabels(Database)

    XGB_model_IDB1_cropped = XGBClassifier(
            gamma=0.7666597170804887,
            learning_rate=0.07803511542647017,
            max_depth=3,
            colsample_bytree=0.6551366523873426,
            scale_pos_weight=2,
            subsample=0.6107674169914816,
            n_estimators=216,
            eval_metric="error",
        )

    #Load Features and carry out concatenation
    featureDic = {}
    xCeption = loadFeatures("xCeption", Database)
    featureDic["xCeption"] = xCeption

    AverageScores = {}
    AverageF1 = {}
    AveragePrec = {}
    AverageRecall = {}

    #Define k cross validation of k = 5, randomize seed value
    kfold = KFold(n_splits=5,shuffle=True,random_state=randint(1,1000))
    #loop through each model
    for modelName,features in featureDic.items():
        scores = []
        f1 = []
        precision = []
        recall = []
        #Train and evaluate
        for train_index, test_index in kfold.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels_encoded[train_index], labels_encoded[test_index]
            eval_set = [(X_train, y_train), (X_test, y_test)]
            XGB_model_IDB1_cropped.fit(X_train, y_train,eval_set=eval_set)
            y_pred = XGB_model_IDB1_cropped.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

            f1.append(f1_score(y_test,y_pred,pos_label=1))
            precision.append(precision_score(y_test,y_pred,pos_label=1))
            recall.append(recall_score(y_test,y_pred,pos_label=1))
            print(y_pred)

            # Save the trained model
            pickle.dump(XGB_model_IDB1_cropped, open(str(modelName) + "_XGB_" + Database + "_Model.pkl", 'wb'))

        #accuracy results
        print("-------Accuracy--------")
        print("{}={}".format(str(modelName),scores))
        print("{}={}".format(str(modelName), np.mean(scores)))
        AverageScores[str(modelName)] = np.mean(scores)
        scores.clear()

        #F1 results
        print("-------F1--------")
        print("{}={}".format(str(modelName), f1))
        print("{}={}".format(str(modelName), np.mean(f1)))
        AverageF1[str(modelName)] = np.mean(f1)
        f1.clear()

        #Prec results
        print("-------Precision--------")
        print("{}={}".format(str(modelName), precision))
        print("{}={}".format(str(modelName), np.mean(precision)))
        AveragePrec[str(modelName)] = np.mean(precision)
        precision.clear()

        #Recall results
        print("-------Recall--------")
        print("{}={}".format(str(modelName), recall))
        print("{}={}".format(str(modelName), np.mean(recall)))
        AverageRecall[str(modelName)] = np.mean(recall)
        recall.clear()


    print(AverageScores)
    print(AverageF1)
    print(AveragePrec)
    print(AverageRecall)

train_all_models("IDB1_cropped")




