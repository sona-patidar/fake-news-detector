import os
import pandas as pd
import string
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load datasets
data_fake = pd.read_csv('Datasets/Fake.csv')
data_true = pd.read_csv('Datasets/True.csv')
print("✅ Data loaded successfully!")

# Label the datasets
data_fake["class"] = 0  # Fake news
data_true["class"] = 1  # Real news

# Remove last 10 rows for testing
data_fake = data_fake[:-10]
data_true = data_true[:-10]

# Merge datasets
data = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data.drop(columns=['title', 'subject', 'date'], inplace=True)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Optimized text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

# Apply preprocessing
data['text'] = data['text'].apply(clean_text)
print("✅ Text preprocessed successfully!")

# Splitting data into training and testing (Stratified to maintain class balance)
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

# Convert text to vectors with optimized TF-IDF
vectorization = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Check class distribution before applying SMOTE
print("🔍 Class distribution before balancing:")
print(y_train.value_counts())

# Apply SMOTE only if the dataset is imbalanced
smote = SMOTE(sampling_strategy=0.7, random_state=42)
try:
    xv_train_resampled, y_train_resampled = smote.fit_resample(xv_train, y_train)
    print("✅ SMOTE applied successfully!")
except ValueError:
    print("⚠️ SMOTE could not be applied. Falling back to Random Undersampling.")
    rus = RandomUnderSampler(random_state=42)
    xv_train_resampled, y_train_resampled = rus.fit_resample(xv_train, y_train)

# Train models with optimizations
LR = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
LR.fit(xv_train_resampled, y_train_resampled)
print("✅ Logistic Regression trained successfully!")

DT = DecisionTreeClassifier(max_depth=10, random_state=42)
DT.fit(xv_train_resampled, y_train_resampled)
print("✅ Decision Tree trained successfully!")

GB = GradientBoostingClassifier(n_estimators=200, random_state=42)
GB.fit(xv_train_resampled, y_train_resampled)
print("✅ Gradient Boosting trained successfully!")

RF = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
RF.fit(xv_train_resampled, y_train_resampled)
print("✅ Random Forest trained successfully!")

# Ensure the models directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Remove old models if directory is not empty
for filename in os.listdir(models_dir):
    file_path = os.path.join(models_dir, filename)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"⚠️ Error deleting {file_path}: {e}")

print("✅ Old models cleared!")

# Save new models
joblib.dump(vectorization, os.path.join(models_dir, 'vectorizer.pkl'))
joblib.dump(LR, os.path.join(models_dir, 'logistic_regression.pkl'))
joblib.dump(DT, os.path.join(models_dir, 'decision_tree.pkl'))
joblib.dump(GB, os.path.join(models_dir, 'gradient_boosting.pkl'))
joblib.dump(RF, os.path.join(models_dir, 'random_forest.pkl'))

print("✅ Models trained and saved successfully!")