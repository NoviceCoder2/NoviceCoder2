import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv(r"C:\Users\sbani\Downloads\Social_Network_Ads.csv")
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
age = float(input("Enter Age: "))
salary = float(input("Enter Estimated Salary: "))
user_data = [[age, salary]]
user_data_scaled = scaler.transform(user_data)
predicted_class = nb_classifier.predict(user_data_scaled)

if predicted_class[0] == 0:
    print("Predicted class: Not Purchased")
else:
    print("Predicted class: Purchased")
