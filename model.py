import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.layers import Activation, Dense, Dropout
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from sklearn.svm import SVC
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier

#Read the dataframe
df1 = pd.read_csv('modified1_file.csv') 

# Encode the months to values
month_to_number = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

# Drop Nan values for discovery_month
df1 = df1.dropna(subset=['discovery_month'])
df1['discovery_month'] = df1['discovery_month'].map(month_to_number) # Encode

categorical_columns = ['discovery_month', 'Vegetation']
numerical_columns = ['latitude', 'longitude', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7']

X_categorical = df1[categorical_columns]
X_numerical = df1[numerical_columns]
X_categorical = X_categorical.astype('category')

#Imppute the missing values for the numerical parameters
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_numerical_imputed = pd.DataFrame(imputer.fit_transform(X_numerical), columns=X_numerical.columns)

#Define the X and y variables
X = pd.concat([X_categorical, X_numerical_imputed], axis=1)
y_fire_class = df1['fire_size_class']
y_putout_time = df1['putout_time']

# Encode the dependant variable
le = LabelEncoder()
y_fire_class = le.fit_transform(y_fire_class)
scaler = MinMaxScaler()
X[['Wind_pre_7', 'Hum_pre_7', 'Temp_pre_7']] = scaler.fit_transform(X[['Wind_pre_7', 'Hum_pre_7', 'Temp_pre_7']])


# Split the dataset for training and testing
X_train, X_test, y_putout_time_train, y_putout_time_test, y_fire_class_train, y_fire_class_test = train_test_split(
    X, y_putout_time, y_fire_class, test_size=0.2, random_state=0)

# Create an object of the classifer class
svm_classifier = SVC(kernel='rbf', C=0.8, gamma='scale')

# Fit the data
svm_classifier.fit(X_train, y_fire_class_train)

# Predict the dependant variable for unknown values of the independant variables
y_pred = svm_classifier.predict(X_test)

#Calculate the Accuracy score
accuracy = accuracy_score(y_fire_class_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the perfomance metrics
classification_report_output = classification_report(y_fire_class_test, y_pred)
print("Classification Report:")
print(classification_report_output)
# joblib.dump(svm_classifier, "SVM.joblib")

##################################################################################
# Neural Network Model

model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],X_train.shape[-1])))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))  # Output layer with 1 unit for regression tasks
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Set epochs and batch size
epochs = 50
batch_size = 64

# fit the model
model.fit(X_train, y_putout_time_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_putout_time_test))