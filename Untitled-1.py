import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

print(train_data.isnull().sum())

X_train = train_data.drop('Цена на арматуру', axis=1)
y_train = train_data['Цена на арматуру']
X_test = test_data.drop('Цена на арматуру', axis=1)

model = CatBoostRegressor(iterations=1000,
                          learning_rate=0.1,
                          depth=6,
                          loss_function='RMSE',
                          verbose=100
                         )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test = test_data['Цена на арматуру']
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

model.save_model('catboost_model.cbm')