import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load datasets
sep = ":::"
train_data = pd.read_csv(r"C:\Users\user\Desktop\train_data.txt", sep=sep, engine='python')
test_data = pd.read_csv(r"C:\Users\user\Desktop\test_data.txt", sep=sep, engine='python')
test_data_solution = pd.read_csv(r"C:\Users\user\Desktop\test_data_solution.txt", sep=sep, engine='python')

# Assign column names
train_data.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
test_data.columns = ['ID', 'TITLE', 'DESCRIPTION']

# Fill missing descriptions with empty strings
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].fillna('')
test_data['DESCRIPTION'] = test_data['DESCRIPTION'].fillna('')

# Encode genre labels
label_encoder = LabelEncoder()
train_data['genre_encoded'] = label_encoder.fit_transform(train_data['GENRE'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['DESCRIPTION'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['DESCRIPTION'])

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, train_data['genre_encoded'], test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Optional)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize models with balanced class weights
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced'),
    "Naïve Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', class_weight='balanced')
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_resampled, y_train_resampled)  # Train with resampled data
    
    # Predictions on validation set
    y_pred = model.predict(X_val)
    
    # Model evaluation (Handling zero-division)
    print(f"\n{model_name} - Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_val, y_pred, zero_division=1))

# Use the best model for final predictions on the test data (You can change the model)
final_model = models["Logistic Regression"]  # Change to "Naïve Bayes" or "SVM" if needed
test_predictions = final_model.predict(X_test_tfidf)

# Convert predicted labels back to original genre names
test_data['Predicted_Genre'] = label_encoder.inverse_transform(test_predictions)

# Save predictions to a file
test_data[['ID', 'Predicted_Genre']].to_csv("predicted_test_genres.csv", index=False)

print("\nPredictions saved to 'predicted_test_genres.csv'")

# ==========================
# PLOTTING ALL GRAPHS IN A SINGLE FIGURE
# ==========================

plt.figure(figsize=(18, 16))

# 1. Movie Genre Distribution (Count plot)
plt.subplot(3, 2, 1)
sns.countplot(data=train_data, x='GENRE', order=train_data['GENRE'].value_counts().index, palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Movie Genre")
plt.ylabel("Count")
plt.title("Movie Genre Distribution in Training Data")

# 2. Histogram with KDE (Black Graph)
plt.subplot(3, 2, 2)
sns.histplot(train_data['GENRE'], kde=True, color='black', bins=30)
plt.xticks(rotation=90)
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Movie Genre Histogram with KDE")

# 3. Histogram (Blue Graph)
plt.subplot(3, 2, 3)
sns.histplot(train_data['GENRE'], color='blue', bins=30)
plt.xticks(rotation=90)
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Movie Genre Histogram")

# 4. Horizontal Bar Plot
plt.subplot(3, 2, 4)
count1 = train_data['GENRE'].value_counts()
sns.barplot(x=count1, y=count1.index, palette="coolwarm", orient='h')
plt.xlabel("Count")
plt.ylabel("Categories")
plt.title("Movie Genre Bar Plot")

# 5. Pie Chart for Genre Distribution
plt.subplot(3, 2, 5)
train_data['GENRE'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
plt.ylabel("")
plt.title("Genre Distribution Pie Chart")

# Display all plots together
plt.tight_layout()
plt.show()
