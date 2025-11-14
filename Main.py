from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from CustomButton import TkinterCustomButton
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import timeit

main = Tk()
main.title("Machine Learning Approaches to Detect DoS and Their Effect on WSNs Lifetime")
main.geometry("1300x1200")

global filename
global X, Y, le, dataset, labels
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, scaler, gb_model

def uploadDataset():
    global filename, labels, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    labels, count = np.unique(dataset['Attack type'], return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (6, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Attack Types")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def preprocessDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, scaler
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset['Attack type'] = pd.Series(le.fit_transform(dataset['Attack type'].astype(str)))#encode all str columns to numeric
    dataset.fillna(0, inplace = True)
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    indices = np.arange(X.shape[0]) #shuffling dataset values
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)#dataset normalization
    text.insert(END,"Processed & Normalzied Dataset values\n\n")
    text.insert(END,str(X)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Train & Test Dataset Split\n\n")
    text.insert(END,"80% records used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records features used to test algorithms : "+str(X_test.shape[0])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict, execution_time):
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n")
    text.insert(END,algorithm+" Execution Time : "+str(execution_time)+"\n\n")

def runKNNLRSVM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    start = timeit.default_timer()
    knn_cls = KNeighborsClassifier(n_neighbors=15) #create KNN object
    knn_cls.fit(X_train, y_train) #train KNN on training data
    predict = knn_cls.predict(X_test)
    end = timeit.default_timer()
    calculateMetrics("KNN", y_test, predict, (end - start))

    start = timeit.default_timer()
    lr_cls = LogisticRegression(solver="liblinear")
    lr_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = lr_cls.predict(X_test) #perform prediction on test data
    end = timeit.default_timer()
    calculateMetrics("Logistic Regression", y_test, predict, (end - start))

    start = timeit.default_timer()
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = svm_cls.predict(X_test) #perform prediction on test data
    end = timeit.default_timer()
    calculateMetrics("SVM", y_test, predict, (end - start))

def runGboost():
    global gb_model
    start = timeit.default_timer()
    gb_cls = GradientBoostingClassifier() 
    gb_cls.fit(X_train, y_train) 
    predict = gb_cls.predict(X_test)
    gb_model = gb_cls
    end = timeit.default_timer()
    calculateMetrics("GBoost", y_test, predict, (end - start))

    start = timeit.default_timer()
    nb_cls = GaussianNB() 
    nb_cls.fit(X_train, y_train) 
    predict = nb_cls.predict(X_test)
    end = timeit.default_timer()
    calculateMetrics("Naive Bayes", y_test, predict, (end - start))
    
def runLSTM():
    start = timeit.default_timer()
    mlp_cls = MLPClassifier() 
    mlp_cls.fit(X_train[0:5000], y_train[0:5000]) 
    predict = mlp_cls.predict(X_test)
    end = timeit.default_timer()
    calculateMetrics("MLP", y_test, predict, (end - start))
    
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)

    start = timeit.default_timer()
    #training LSTM algorithm
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train1.shape[1], X_train1.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.5))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train1, y_train1, batch_size = 16, epochs = 10, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    #perform prediction on test data   
    predict = lstm_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    end = timeit.default_timer()
    calculateMetrics("LSTM Model", y_test1, predict, (end - start))#calculate accuracy and other metrics

def graph():
    #comparison graph between all algorithms
    df = pd.DataFrame([['KNN','Accuracy',accuracy[0]],['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','FSCORE',fscore[0]],
                       ['Logistic Regression','Accuracy',accuracy[1]],['Logistic Regression','Precision',precision[1]],['Logistic Regression','Recall',recall[1]],['Logistic Regression','FSCORE',fscore[1]],
                       ['SVM','Accuracy',accuracy[2]],['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','FSCORE',fscore[2]],
                       ['GBoost','Accuracy',accuracy[3]],['GBoost','Precision',precision[3]],['GBoost','Recall',recall[3]],['GBoost','FSCORE',fscore[3]],
                       ['Naive Bayes','Accuracy',accuracy[4]],['Naive Bayes','Precision',precision[4]],['Naive Bayes','Recall',recall[4]],['Naive Bayes','FSCORE',fscore[4]],
                       ['MLP','Accuracy',accuracy[5]],['MLP','Precision',precision[5]],['MLP','Recall',recall[5]],['MLP','FSCORE',fscore[5]],
                       ['LSTM','Accuracy',accuracy[6]],['LSTM','Precision',precision[6]],['LSTM','Recall',recall[6]],['LSTM','FSCORE',fscore[6]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def predict():
    global gb_model, scaler, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(filename)
    data = dataset.values
    X = scaler.transform(data)
    predict = gb_model.predict(X)
    for i in range(len(predict)):        
        text.insert(END,"Test Data = "+str(data[i])+" Predicted As ====> "+labels[int(predict[i])]+"\n\n")

font = ('times', 15, 'bold')
title = Label(main, text='Machine Learning Approaches to Detect DoS and Their Effect on WSNs Lifetime')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = TkinterCustomButton(text="Upload WSN DOD Dataset", width=300, corner_radius=5, command=uploadDataset)
uploadButton.place(x=50,y=150)

preprocessButton = TkinterCustomButton(text="Preprocess Dataset", width=300, corner_radius=5, command=preprocessDataset)
preprocessButton.place(x=370,y=150)

knnButton = TkinterCustomButton(text="Run KNN, LR & SVM Algorithms", width=300, corner_radius=5, command=runKNNLRSVM)
knnButton.place(x=690,y=150)

gboostButton = TkinterCustomButton(text="Run GBoost & Naive Bayes Algorithms", width=300, corner_radius=5, command=runGboost)
gboostButton.place(x=50,y=200)

lstmButton = TkinterCustomButton(text="Run LSTM & MLP Algorithms", width=300, corner_radius=5, command=runLSTM)
lstmButton.place(x=370,y=200)

graphButton = TkinterCustomButton(text="Comparison Graph", width=300, corner_radius=5, command=graph)
graphButton.place(x=690,y=200)

predictButton = TkinterCustomButton(text="Detect Attack from Test Data", width=300, corner_radius=5, command=predict)
predictButton.place(x=50,y=250)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
