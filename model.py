df1 = pd.read_csv('modified1_file.csv')

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

df1 = df1.dropna(subset=['discovery_month'])
df1['discovery_month'] = df1['discovery_month'].map(month_to_number)

categorical_columns = ['discovery_month', 'Vegetation']
numerical_columns = ['latitude', 'longitude', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7']

X_categorical = df1[categorical_columns]
X_numerical = df1[numerical_columns]
X_categorical = X_categorical.astype('category')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_numerical_imputed = pd.DataFrame(imputer.fit_transform(X_numerical), columns=X_numerical.columns)


X = pd.concat([X_categorical, X_numerical_imputed], axis=1)
y_fire_class = df1['fire_size_class']
y_putout_time = df1['putout_time']

le = LabelEncoder()
y_fire_class = le.fit_transform(y_fire_class)
scaler = MinMaxScaler()
X[['Wind_pre_7', 'Hum_pre_7', 'Temp_pre_7']] = scaler.fit_transform(X[['Wind_pre_7', 'Hum_pre_7', 'Temp_pre_7']])

X_train, X_test, y_putout_time_train, y_putout_time_test, y_fire_class_train, y_fire_class_test = train_test_split(
    X, y_putout_time, y_fire_class, test_size=0.2, random_state=0)

svm_classifier = SVC(kernel='rbf', C=0.8, gamma='scale')

svm_classifier.fit(X_train, y_fire_class_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_fire_class_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

classification_report_output = classification_report(y_fire_class_test, y_pred)
print("Classification Report:")
print(classification_report_output)
# joblib.dump(svm_classifier, "SVM.joblib")