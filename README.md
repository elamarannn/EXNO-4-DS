# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
Name : Elamaran S E

Reg.no ; 212222230036
```
import pandas as pd
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df
```
![388438307-a66652f9-c881-4a86-9d60-f92810b15c79](https://github.com/user-attachments/assets/b0a85a45-14b4-4912-a65f-d76381fc4705)

```
df.head()
```
![388438564-da322bb8-5154-4a88-956d-6b2ddc277842](https://github.com/user-attachments/assets/ae8feb2d-807c-43a4-b2ff-e4d69dd7ef3f)

```
df.dropna()
```
![388500297-be732e5d-f3d4-43d3-98cd-e3880852c0f4](https://github.com/user-attachments/assets/af208bf5-46fc-4f5f-91a6-f3ad15f2b0e9)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![388500631-f637300f-cced-442c-a903-39645a36dbfb](https://github.com/user-attachments/assets/8bbf2939-2036-4441-aa28-36a868b4ca03)
```
from sklearn.preprocessing import MinMaxScaler
```
```
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])fr
```
```
df.head(10)
```
![388502238-1365399a-aa4c-4cb7-a4b6-796831621969](https://github.com/user-attachments/assets/5b4432ce-ba90-406e-bc23-49d9d8b98a2b)

```
df1=pd.read_csv("bmi.csv")
```
```
df2=pd.read_csv("bmi.csv")
```
```
df3=pd.read_csv("bmi.csv")
```
```
df4=pd.read_csv("bmi.csv")
```
```
df5=pd.read_csv("bmi.csv")
```
```
df1

```
![388504056-a7322a98-3297-41b6-a5e8-595c91dc8b4d](https://github.com/user-attachments/assets/638d1a02-12be-449f-8d7c-2a98b81564fc)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![388506622-6ce0dc2e-72ee-4283-ab1a-61e105b7b596](https://github.com/user-attachments/assets/941ff51a-05c5-4fe5-b6bb-c09ef6eb0188)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
```
```
df2
```
![388506982-d5705138-76ec-4a7f-ab0a-04f1e473fdc6](https://github.com/user-attachments/assets/3408f9ab-5314-4b4c-8440-49a76c389e8c)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df3
```
![388507343-074d8e0c-80fa-47c0-bac6-7353f1b39ab7](https://github.com/user-attachments/assets/05b808c7-6c42-4a38-8cd6-c6579038ef21)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df4
```
![388507670-de4deeba-b8ab-4c5d-8818-6768a02c8543](https://github.com/user-attachments/assets/32f4bb7e-d8d0-480e-b156-5c2a63421182)

```
import seaborn as sns
```
```
feature selection 
import pandas as pd

import numpy as np 
import seaborn as sns
```
```
import seaborn as sns
```
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
```
```
data=pd.read_csv("titanic_dataset.csv")
data
```
![388511413-c0330f51-a459-4e63-ba8f-8356dc929c69](https://github.com/user-attachments/assets/d6ea730b-9c1e-439e-accb-75142697f517)

```
data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
```
```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
```
```
data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes
```
```
 data
```
![388515515-2a598b08-3121-478e-a22b-c7c0986f79c3](https://github.com/user-attachments/assets/5973b09b-afbc-421f-824a-fbfe4317c8ae)

```

k=5
selector=SelectKBest(score_func=chi2, k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
```
```
x_encoded =pd.get_dummies(x)
selector=SelectKBest(score_func=chi2, k=5)
x_new = selector.fit_transform(x_encoded,y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected_Feature:")
print(selected_features)
```
![388516439-317c121c-abf4-4dc7-a14d-0f3e899e50b4](https://github.com/user-attachments/assets/0dca74cf-753b-42d8-8284-f5d02e7f7b31)

```
selector=SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![388516903-70e1b6b3-e6a9-4b0c-abf6-e257e8dff14f](https://github.com/user-attachments/assets/cbc22ad3-a5ec-40a4-9dac-749f93ab3e33)

```
selector=SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![388521115-1814553c-c56f-42a4-9416-589884be81bd](https://github.com/user-attachments/assets/010f1511-ba41-4b59-8924-e18d89f3dd0b)
 
```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```
![388517155-412dcc3f-3af8-4678-9da0-1a354173f174](https://github.com/user-attachments/assets/41ee65b0-1c33-4fd3-bd43-baa6fe18339b)

```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.1
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![388517612-afe8c286-877d-41f4-9392-3aa16803f600](https://github.com/user-attachments/assets/7716bd98-6b9b-402c-9538-cf5f53012960)

```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.15
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![388517967-dca07307-d7fc-4546-8614-fadbeb133ca0](https://github.com/user-attachments/assets/7246f00d-bae5-4402-8344-1c3365e8c6c2)



# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
