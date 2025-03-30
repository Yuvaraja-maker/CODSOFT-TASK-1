import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import zipfile

##Load the fraudTrain and fraaudTest datasets

zip_path =r"C:\Users\user\Downloads\credit card fraud detection.zip"
csv_files = ["fraudTrain.csv","fraudTest.csv"]

with zipfile.ZipFile(zip_path, 'r') as z:
    # Read fraudTrain.csv
    with z.open(csv_files[0]) as file1:
        data_train = pd.read_csv(file1)
    # Read fraudTest.csv
    with z.open(csv_files[1]) as file2:
        data_test = pd.read_csv(file2)

# To read the fraudTrain data
print("Train Data:")
print(data_train)
##print(data_train.head())
##print(data_train.tail())
##print(data_train.shape)
##print(data_train.size)
##print(data_train.describe())
data_train.drop_duplicates(inplace=True)
print("Column names:", data_train.columns.tolist()) 

# To read the fraudTest data
print("\nTest Data:")
print(data_test)
##print(data_test.head())
##print(data_test.tail())
##print(data_test.shape)
##print(data_test.size)
##print(data_test.describe())
data_test.drop_duplicates(inplace=True)
print("Column names:", data_test.columns.tolist())


#combine training and testing data for encoding consistency
combined_data = pd.concat([data_train,data_test],axis=0)

#Extract revelant features from the "trans_date_trans_time" column

def extract_datetime_features(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
    return df.drop(columns=['trans_date_trans_time'])
tqdm.pandas()
combined_data = extract_datetime_features(combined_data)

# Identify categorical columns (excluding target variable)
categorical_cols = combined_data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

###Encode categorical variables
##label_encoders = {}
##for col in ['category','gender']:
##    le=LabelEncoder()
##    combined_data[col] = le.fit_transform(combined_data[col])
##    label_encoders[col] = le

# Separate features and target variable
target_column = 'is_fraud'
X = combined_data.drop(columns=[target_column, 'cc_num', 'first', 'last', 'street', 'dob', 'trans_num'])  # Remove unnecessary columns
y = combined_data[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalize numerical features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train and evaluate multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=50,n_jobs=-1, random_state=42)
       }

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))




