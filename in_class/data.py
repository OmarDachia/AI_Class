
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('in_class/Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

X_train, X_test, Y_train, Y_test = train_test_split( x , y , test_size = 0.2, random_state = 0)
sc_X = StandardScaler()

# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

print(X_train)