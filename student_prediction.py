import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Study_Hours': [2,3,4,5,6,7,8,9,10],
    'Attendance': [50,55,60,65,70,75,80,85,90],
    'Marks': [35,40,45,55,60,70,78,85,92]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

study = int(input("Enter Study Hours: "))
attend = int(input("Enter Attendance: "))

prediction = model.predict([[study, attend]])
print("Predicted Marks:", int(prediction[0]))
