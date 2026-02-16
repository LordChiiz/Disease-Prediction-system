# to train a model with the dataset to be used in diagnosis
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier #forest to handle binary symptom data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib #for me to save the trained model so that i dont have to retrain it every time

print("Loading dataset...")#oh yes, very important :D
data = pd.read_csv('Training.csv')

X = data.drop('prognosis', axis=1)
y = data['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the AI model... (This might take a moment)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Trained Successfully!")
print(f"Accuracy Score: {accuracy * 100:.2f}%") #:.2f formats to 2 decimal places 

joblib.dump(model, 'disease_model.pkl')

symptom_columns = list(X.columns)
joblib.dump(symptom_columns, 'symptoms_list.pkl')

print("Files 'disease_model.pkl' and 'symptoms_list.pkl' saved!")