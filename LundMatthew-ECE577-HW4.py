#Matthew Lund
#mtlund@wpi.edu
#ECE 577 Project 4 PRNG Classifier

import numpy as np
import os
import sys

#ML Algorithm Imports
from sklearn import svm, neighbors, tree
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import to_categorical
import lightgbm as lgb

#Vectors and Data arrrays for each section of bits

def load_data(bit_num, dataset_path):
    base_path = dataset_path
    
    bit_dir = f"{bit_num}-bit"
    
    train_path = os.path.join(base_path, bit_dir, f"train_{bit_num}bit.txt")
    validation_path = os.path.join(base_path, bit_dir, f"val_{bit_num}bit.txt")
    test_path = os.path.join(base_path, bit_dir, f"test_{bit_num}bit.txt")

    #Invalid Raise Handling based on contents of files given

    train_data = np.genfromtxt(train_path, delimiter=',', invalid_raise=False)
    validation_data = np.genfromtxt(validation_path, delimiter=',', invalid_raise=False)
    test_data = np.genfromtxt(test_path, delimiter=',', invalid_raise=False)
    
    # Handle missing labels by setting them to 0
    train_data = np.nan_to_num(train_data, nan=0)
    validation_data = np.nan_to_num(validation_data, nan=0)
    test_data = np.nan_to_num(test_data, nan=0)

    train_label = train_data[:, -1]
    validation_label = validation_data[:, -1]
    test_label = test_data[:, -1]

    train_data_vectors = train_data[:, :-1]
    validation_data_vectors = validation_data[:, :-1]
    test_data_vectors = test_data[:, :-1]

    vector_size = train_data_vectors.shape[1]
    output_path = os.path.join(base_path, bit_dir)
    return output_path, train_data_vectors, train_label, validation_data_vectors, validation_label, test_data_vectors, test_label, vector_size

#Run the models and get accuracy numbers
    
def models_run(bit_num, dataset_path):
    output_path, train_data, train_label, validation_data, validation_label, test_data, test_label, vector_size = load_data(bit_num, dataset_path)
    print("Testing", bit_num,"- bit Data")
   
    #All data printed to output files:

    output_file = os.path.join(output_path, f"results_{bit_num}bit.txt")
    
    with open(output_file, "w") as file:

        #SVM
        clf_svm = svm.SVC(kernel ='linear', C = 1.0)
        clf_svm.fit(train_data, train_label)
        print("SVM training accuracy:", clf_svm.score(train_data, train_label), file=file)
        print("SVM test accuracy:", clf_svm.score(test_data, test_label), file=file)
        print("SVM validation accuracy:", clf_svm.score(validation_data, validation_label), file=file)

        #kNN
        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5)

        clf_knn.fit(train_data, train_label)
        print("kNN training accuracy:", clf_knn.score(train_data, train_label), file=file)
        print("kNN test accuracy:", clf_knn.score(test_data, test_label), file=file)
        print("kNN validation accuracy:", clf_knn.score(validation_data, validation_label), file=file)
        
        #Dec. Tree
        clf_dt = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1)
        clf_dt.fit(train_data, train_label)
        print("Decision Tree training accuracy:", clf_dt.score(train_data, train_label), file=file)
        print("Decision Tree test accuracy:", clf_dt.score(test_data, test_label), file=file)
        print("Decision Tree validation accuracy:", clf_dt.score(validation_data, validation_label), file=file)

        #LightGBM
        clf_lgb = lgb.LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)
        clf_lgb.fit(train_data, train_label)
        print("LightGBM training accuracy:", clf_lgb.score(train_data, train_label), file=file)
        print("LightGBM test accuracy:", clf_lgb.score(test_data, test_label), file=file)
        print("LightGBM validation accuracy:", clf_lgb.score(validation_data, validation_label), file=file)
        
        
        #One-Hot Encoding for CNN
        train_label -= 1
        test_label -= 1
        validation_label -= 1
        train_label = to_categorical(train_label, num_classes=8)
        test_label = to_categorical(test_label, num_classes=8)
        validation_label = to_categorical(validation_label, num_classes=8)

        #CNN
        model_cnn = Sequential()
        model_cnn.add(Conv1D(64, 3, activation='relu', input_shape=(vector_size, 1)))
        model_cnn.add(MaxPooling1D(2))
        model_cnn.add(Conv1D(128, 3, activation='relu'))
        model_cnn.add(GlobalAveragePooling1D())
        model_cnn.add(Dense(8, activation='softmax'))  #8 classes

        model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_cnn.fit(train_data.reshape(train_data.shape[0], train_data.shape[1], 1), train_label,
                    validation_data=(validation_data.reshape(validation_data.shape[0], validation_data.shape[1], 1), validation_label),
                    epochs=20, batch_size=64)

        print("CNN training accuracy:", model_cnn.evaluate(train_data.reshape(train_data.shape[0], train_data.shape[1], 1), train_label)[1], file=file)
        print("CNN test accuracy:", model_cnn.evaluate(test_data.reshape(test_data.shape[0], test_data.shape[1], 1), test_label)[1], file=file)
        print("CNN validation accuracy:", model_cnn.evaluate(validation_data.reshape(validation_data.shape[0], validation_data.shape[1], 1), validation_label)[1], file=file)

    print("Testing Completed. Results can be found in" , output_file)


#Run from Console
if __name__ == "__main__":
    bit_num = sys.argv[1]
    dataset_path = sys.argv[2]

    #Match this number to filepath active for accurate printing results
    models_run(bit_num, dataset_path)