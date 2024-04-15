#Matthew Lund
#mtlund@wpi.edu
#ECE 577 Project 4 PRNG Classifier

import numpy as np
import os
import sys
from sklearn import svm, neighbors, tree
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, SimpleRNN, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import to_categorical

#Vectors and Data arrrays for each section of bits
def load_data():
    
    #Initialize vectors
    train_vector = validation_vector = test_vector = None, None, None  # Initialize variables
    
    # Depending on what # of bits looking to test

    #32 Bits
    train_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\32-bit\\train_32bit.txt', delimiter=',')
    validation_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\32-bit\\val_32bit.txt', delimiter=',')
    test_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\32-bit\\test_32bit.txt', delimiter=',')

    #64 Bits
    # train_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\64-bit\\train_64bit.txt', delimiter=',')
    # validation_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\64-bit\\val_64bit.txt', delimiter=',')
    # test_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\64-bit\\test_64bit.txt', delimiter=',')

    #128 Bits
    # train_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\128-bit\\train_128bit.txt', delimiter=',')
    # validation_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\128-bit\\val_128bit.txt', delimiter=',')
    # test_vector = np.loadtxt('C:\\Users\\Matthew\\OneDrive\\Documents\\ECE577\\ECE-577-Proj4\\PRNG-Dataset\\128-bit\\test_128bit.txt', delimiter=',')
    
    train_data = train_vector[:,:-1]
    train_label = train_vector[:,-1]

    validation_data = validation_vector[:,:-1]
    validation_label = validation_vector[:,-1]

    test_data = test_vector[:,:-1]
    test_label = test_vector[:,-1]

    vector_size = train_data.shape[1]
    return train_data, train_label, validation_data, validation_label, test_data, test_label, vector_size

#Run the models and get accuracy numbers
    
def models_run(bit_num):
    print("Testing", bit_num,"- bit Data")
    train_data, train_label, validation_data, validation_label, test_data, test_label, vector_size = load_data()
    
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
    
    #One-Hot Encoding for RNN and CNN
    train_label -= 1
    test_label -= 1
    validation_label -= 1
    train_label = to_categorical(train_label, num_classes=8)
    test_label = to_categorical(test_label, num_classes=8)
    validation_label = to_categorical(validation_label, num_classes=8)

    #RNN
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(units=64, activation='relu', input_shape=(vector_size, 1)))
    model_rnn.add(Dense(units=8, activation='softmax'))  # Assuming 8 classes
    model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_rnn.fit(train_data.reshape(train_data.shape[0], train_data.shape[1], 1), train_label, epochs=10, batch_size=32)
    print("RNN accuracy:", model_rnn.evaluate(test_data.reshape(test_data.shape[0], test_data.shape[1], 1), test_label)[1])   
    
    #CNN
    model_cnn = Sequential()
    model_cnn.add(Conv1D(32, 3, activation='relu', input_shape=(vector_size, 1)))
    model_cnn.add(MaxPooling1D(2))
    model_cnn.add(Conv1D(64, 3, activation='relu'))
    model_cnn.add(GlobalAveragePooling1D())
    model_cnn.add(Dense(8, activation='softmax'))  # Assuming 8 classes

    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_cnn.fit(train_data.reshape(train_data.shape[0], train_data.shape[1], 1), train_label,
                validation_data=(validation_data.reshape(validation_data.shape[0], validation_data.shape[1], 1), validation_label),
                epochs=10, batch_size=32)

    print("CNN accuracy:", model_cnn.evaluate(test_data.reshape(test_data.shape[0], test_data.shape[1], 1), test_label)[1])

#Run from Console
if __name__ == "__main__":
    bit_num = sys.argv[1]
    #Match this number to filepath active for accurate printing results
    models_run(bit_num)