#Matthew Lund
#mtlund@wpi.edu
#ECE 577 Project 4 PRNG Classifier

import numpy as np
import os
import sys
from sklearn import svm, neighbors, tree
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.utils import to_categorical

#Vectors and Data arrrays for each section of bits
def load_data(bit_num):

    #Depending on what # of bits looking to test
    if bit_num == 32:
        train_vector=np.loadtxt()
        validation_vector=np.loadtxt()
        test_vector=np.loadtxt()
    elif bit_num == 64:     
        train_vector=np.loadtxt()
        validation_vector=np.loadtxt()
        test_vector=np.loadtxt()
    elif bit_num == 128:
        train_vector=np.loadtxt()
        validation_vector=np.loadtxt()
        test_vector=np.loadtxt()
    
    train_data =train_vector[:,:-1]
    train_label =train_vector[:,-1]

    validation_data = validation_vector[:,:-1]
    validation_label = validation_vector[:,-1]

    test_data = test_vector[:,:-1]
    
    test_label = test_vector[:,-1]

    vector_size=train_data.shape[1]
    yield train_data, train_label, validation_data, validation_label, test_data, test_label, vector_size

def models_run(bit_num):
    print("Testing {bit_num}-bit Data")
    for train_data, train_label, validation_data, validation_label, test_data, test_label, vector_size in load_data(bit_num):
        #SVM
        clf_svm = svm.SVC()
        clf_svm.fit(train_data, train_label)
        print("SVM accuracy:", clf_svm.score(test_data, test_label))

        #kNN
        clf_knn = neighbors.KNeighborsClassifier()
        clf_knn.fit(train_data, train_label)
        print("kNN accuracy:", clf_knn.score(test_data, test_label))

        #Dec. Tree
        clf_dt = tree.DecisionTreeClassifier()
        clf_dt.fit(train_data, train_label)
        print("Decision Tree accuracy:", clf_dt.score(test_data, test_label))
        
        #RNN
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=vector_size))
        model.add(Dense(units=8, activation='softmax'))  # Assuming 8 classes
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train_data, train_label, epochs=10, batch_size=32)
        print("RNN accuracy:", model.evaluate(test_data, test_label)[1])

        #CNN
        # Reshape the data to be 3-dimensional (sample_number, vector_size, 1) as required by Conv1D
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        validation_data = validation_data.reshape(validation_data.shape[0], validation_data.shape[1], 1)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

        # One-hot encode the labels
        train_label = to_categorical(train_label)
        validation_label = to_categorical(validation_label)
        test_label = to_categorical(test_label)

        # Create the model
        model = Sequential()
        model.add(Conv1D(32, 2, activation='relu', input_shape=(vector_size, 1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(8, activation='softmax'))  # Assuming 8 classes

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(train_data, train_label, validation_data=(validation_data, validation_label), epochs=10, batch_size=32)

        # Evaluate the model
        print("CNN accuracy:", model.evaluate(test_data, test_label)[1])

#Run from Console
if __name__ == "__main__":
    bit_num = sys.argv[1]
    models_run(bit_num)