import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import time
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pca import *
import csv as csv

#This function reshapes the pixel arrays to digits and plots them
def plot_digit(X):
    counter =1
    for i in range(1,4):
        for j in range(1,6):
            plt.subplot(3,5,counter)
            plt.imshow(X[(i-1)*4000 + j].reshape((28,28)),cmap=cm.Greys_r)
            plt.axis('off')
            counter+=1
    plt.show()

def load():
    df = pd.read_csv("dataset/train.csv",delimiter=",",header=0)
    # print df.describe()
    # print df.head()
    return df

def model(X,Y):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
    pipeline = Pipeline([
        ('clf',SVC(kernel='rbf',C =3,gamma = 0.01))
    ])
    parameters = {
        # 'clf__gamma':(0.01,0.03,0.1,0.3,1),
        # 'clf__C':(0.1,0.3,1,3,10,30),
    }
    grid_search = GridSearchCV(pipeline,parameters,n_jobs=2,verbose=1,scoring="accuracy")
    grid_search.fit(X,Y)
    print "Best score on model:",grid_search.best_score_
    print "parameters:"
    best_parameter = grid_search.best_estimator_.get_params()
    print best_parameter
    # pred = grid_search.predict(X_test)
    # print classification_report(Y_test,pred)

    df = pd.read_csv("dataset/test.csv",delimiter=",",header=0)

    df = np.array(df)
    x_test = df
    x_test =  x_test/255.0 * 2 -1
    # reduced_X_test = pca(X,Y)
    pred = grid_search.predict(x_test).tolist()
    ids = list(range(1, 28001))

    with open("results.csv", "wb") as predictions_file:
        # predictions_file = open("myfirstforest.csv", "wb")
        open_file_object = csv.writer(predictions_file, delimiter=',')
        open_file_object.writerow(["ImageId","Label"])
        open_file_object.writerows(zip(ids,pred))
        # predictions_file.close()
        print "Done"

if __name__ == '__main__':
    start_time = time.time()
    df = load()
    print("--- %sload time seconds ---" % (time.time() - start_time))
    df = np.array(df)
    X = df[:,1:]
    Y = df[:,0]
    # plot_digit(X)
    X = X/255.0 * 2 -1
    # reduced_X = pca(X,Y)
    # model(reduced_X,Y)

    model(X,Y)
    # Based on the sample codes in Kaggle and the kernels, this model was the reference for me to develop a code in Matlb.
