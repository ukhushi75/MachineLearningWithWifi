import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier

# As given in the mail by Amin
def Dataset_Loader(root_dir):
    data_list = glob.glob(root_dir+'\\CSI_Dataset\\data\\*.csv')
    label_list = glob.glob(root_dir+'\\CSI_Dataset\\label\\*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('\\')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = data_norm
    for label_dir in label_list:
        label_name = label_dir.split('\\')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = label
    return WiFi_data

CSI_data = Dataset_Loader('.')

x_train = CSI_data['x_train']
x_test  = CSI_data['x_test']
y_train = CSI_data['y_train']
y_test = CSI_data['y_test']

# Reshaping the Data as hinted in F part for 3D to 2D arrays
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)  # (3977, 250*90)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)  # (500, 250*90)

# Part E KNN classification method 

# Define different values of K
k_values = [15, 10, 5, 2, 1]

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train_reshaped)
X_test = scaler.transform(x_test_reshaped)

time_array = []
acc_array = []

knn_p4 = []

for k in k_values:
    # Initialize KNN classifier referenced from the link provided
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Time complexity graph training times requirements
    start_time = time.time()
    
    # Train the model
    knn.fit(X_train, y_train)
    
    #Time taken
    taken_time = time.time() - start_time
    time_array.append(taken_time)
    
    # Make predictions
    predictions = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    acc_array.append(accuracy)    
    print(f"K = {k}, Accuracy: {accuracy}, Time elapsed: {taken_time:.4f} s")
    knn_p4.append(knn)
    
# Plotting accuracy
plt.figure("KNN",figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, acc_array, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')

# Plotting time complexity
plt.subplot(1, 2, 2)
plt.plot(k_values, time_array, marker='o', linestyle='-', color='r')
plt.title('Time Complexity vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Time Taken (seconds)')

plt.tight_layout()
plt.show(block=False)

print("\n")
# Part E SVM Method
# Create a svm Classifier
clf = svm.SVC() # Linear Kernel

# Training the data
clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred= clf.predict(X_test)

#Calculating accuracy
print("SVM: ")
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\n")
# Part E Random Forest Classification

estimators = [5,10,20]
times_list = []
accu_array_rf = []

RF_p4 = []

print("Random Forrest: ")

for est in estimators:
    
    rf = RandomForestClassifier(n_estimators=est)
    
    fit_time = time.time()
    
    rf.fit(X_train,y_train)
    
    time_rf = time.time() - fit_time
    times_list.append(time_rf)
    
    predicts = rf.predict(X_test)
    
    accu = accuracy_score(y_test,predicts)
    accu_array_rf.append(accu)
    
    print(f"Estimator = {est}, Accuracy: {accu}, Time elapsed: {time_rf:.4f} s")
    RF_p4.append(rf)
    
# Plotting accuracy
plt.figure("Random Forest",figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(estimators, accu_array_rf, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. estimator Value')
plt.xlabel('Estimator Value')
plt.ylabel('Accuracy')

# Plotting time complexity
plt.subplot(1, 2, 2)
plt.plot(estimators, times_list, marker='o', linestyle='-', color='r')
plt.title('Time Complexity vs. Estimators')
plt.xlabel('Estimator Value')
plt.ylabel('Time Taken (seconds)')

plt.tight_layout()
plt.show(block = False)

# Part E Confusion Matrix
# Top Performing Cases
KNN_index = acc_array.index(max(acc_array))
#SVM classifier doesn't have an array in this case
RF_index = accu_array_rf.index(max(accu_array_rf))

KNN_re = knn_p4[KNN_index]

RF_re = RF_p4[RF_index]

titles_options = [
    ('KNN CLassifier',KNN_re),
    ('SVM Classifier',clf),
    ('Random Forest Classifier',RF_re)
]

for title, classifier in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels= np.unique(y_test),
        cmap = plt.cm.Blues,
        normalize='true',
    )
    disp.ax_.set_title(title)

plt.show()
