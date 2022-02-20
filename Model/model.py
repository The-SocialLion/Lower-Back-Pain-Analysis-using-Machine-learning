import pandas as pd # Library to process the dataframe
import numpy as np # Library to handle with numpy arrays
import warnings # Library that handles all the types of warnings during execution
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore") # Ignore all the warnings

df=pd.read_csv('F:\python\projects\Lower Back pain Prediction\Dataset_spine.csv')
df=df.drop(columns=['Unnamed: 13'])
df=df.dropna(how='any')
# Expanding the dataset
lst=[df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df,df]
Df=pd.concat(lst)
Df=Df.reset_index(drop=True)
# Label Encoder
le=LabelEncoder()
Df['Class_att']=le.fit_transform(Df['Class_att'])
# Feature separation
X=Df.iloc[:,:-1].values
y=Df.iloc[:,-1].values
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Classification model
gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train,y_train)
#  model file
joblib.dump(gbc, "gbc.pkl")
# Printing Vlaidation Results
print(gbc.score(X_test,y_test))

