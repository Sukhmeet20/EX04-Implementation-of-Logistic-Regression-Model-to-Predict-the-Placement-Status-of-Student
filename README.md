# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset and perform any necessary preprocessing, such as handling missing values
and encoding categorical variables.
2. Initialize the logistic regression model and train it using the training data.
3. Use the trained model to predict the placement status for the test set.
4. Evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sukhmeet Kaur G
RegisterNumber: 2305001032
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
datal-data.copy()
datal.head()
datal datal.drop(['s1_no', 'salary'], axis-1) datal
from sklearn.preprocessing import LabelEncoder
le-LabelEncoder()
datal ["gender"]=le.fit_transform(data1["gender"]).
data1["ssc_b"]=le.fit_transform(datal ["ssc_b"])
datal["hsc_b"]=le.fit_transform(data1["hsc_b"]) datal["hsc_s"]=le.fit_transform(datal["hsc_s"])
data1["degree_t"]=le.fit_transform(datal ["degree_t"])
datal ["workex"]=le.fit_transform(data1["workex"]) datal ["specialisation"]=le.fit_transform(datal["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
datal
x=datal.iloc[:,: -1]
y-datal.iloc[:,-1]
y
from sklearn.model selection import train test split
x_train,x_test,y_train,y_test-train_test_split(x,y,test_size=8.2, random_state=8)
from sklearn.linear_model import LogisticRegression
model LogisticRegression (solver="liblinear")
model.fit(x_train,y_train)
y_pred-model.predict(x_test)
y_pred, x_test
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
accuracy-accuracy_score(y_test,y_pred)
confusion-confusion_matrix(y_test,y_pred)
cr-classification_report(y_test,y_pred)
print("Accuracy score:", accuracy) print("\nConfusion matrix:\n", confusion)
print("\nClassification report:\n",cr)
from sklearn import metrics
cm_display-metrics.ConfusionMatrixDisplay (confusion_matrix-confusion, display_labels=[Tru
cm_display.plot()
```

## Output:
![Screenshot 2024-10-17 110400](https://github.com/user-attachments/assets/1e733810-9e57-483c-848f-d53875a3980a)
![Screenshot 2024-10-17 110410](https://github.com/user-attachments/assets/d0d7d834-6e70-4e88-b1b2-723b1b252a02)
![Screenshot 2024-10-17 110425](https://github.com/user-attachments/assets/bfe907ca-f476-4bb6-9dd5-ae552324d872)
![Screenshot 2024-10-17 110438](https://github.com/user-attachments/assets/e6680c28-7fce-407f-be5a-757565818362)
![Screenshot 2024-10-17 110445](https://github.com/user-attachments/assets/d8ea2711-7126-408e-91a4-6f5ba0c58ef3)
![Screenshot 2024-10-17 110454](https://github.com/user-attachments/assets/9fb92ca3-d317-4074-8356-3fd57540d149)
![Screenshot 2024-10-17 110505](https://github.com/user-attachments/assets/b930d985-7026-484e-b687-a4c2d11229e5)
![Screenshot 2024-10-17 110514](https://github.com/user-attachments/assets/45b34539-f08c-4d4a-997b-f3061a8a8ff7)
![Screenshot 2024-10-17 110555](https://github.com/user-attachments/assets/7396170b-3178-4e7e-9c17-65c2625326f3)
![Screenshot 2024-10-17 110606](https://github.com/user-attachments/assets/8b61b536-e1f4-4f47-8710-e885a712ba51)
![Screenshot 2024-10-17 105030](https://github.com/user-attachments/assets/f70eb1db-5ea1-4edd-92e6-a16fb33b8492)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
