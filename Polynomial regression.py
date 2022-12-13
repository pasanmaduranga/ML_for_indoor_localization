import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

data_frame = pd.read_csv("Data.csv") 


input_data = data_frame[['RSSI1', 'RSSI2', 'RSSI3']]
label_x = data_frame[['X_Actual']]
label_y = data_frame[['Y_Actual']]


# Split data into train and test to verify accuracy after fitting the model. 
input_x_train, input_x_test, label_x_train, label_x_test = train_test_split(input_data, label_x, test_size=0.3, random_state=5)
input_y_train, input_y_test, label_y_train, label_y_test = train_test_split(input_data, label_y, test_size=0.3, random_state=5)


# Build PR model
poly_reg = PolynomialFeatures(degree=4)

LR_x_model = linear_model.LinearRegression()
LR_y_model = linear_model.LinearRegression()


# Training
LR_x_model.fit(poly_reg.fit_transform(input_x_train), label_x_train)
LR_y_model.fit(poly_reg.fit_transform(input_y_train), label_y_train)

# Prediction
predict_x_train = LR_x_model.predict(poly_reg.fit_transform(input_x_train))
predict_x_test = LR_x_model.predict(poly_reg.fit_transform(input_x_test))

predict_y_train = LR_y_model.predict(poly_reg.fit_transform(input_y_train))
predict_y_test = LR_y_model.predict(poly_reg.fit_transform(input_y_test))


# Training and testing accuraciss
print('Training MSE X', mean_squared_error(label_x_train, predict_x_train))
print('Testing MSE X', mean_squared_error(label_x_test, predict_x_test))

print('Training MSE Y', mean_squared_error(label_y_train, predict_y_train))
print('Testing MSE Y', mean_squared_error(label_y_test, predict_y_test))


# Dataset accuracy
x_prediction = LR_x_model.predict(poly_reg.fit_transform(input_data))
y_prediction = LR_y_model.predict(poly_reg.fit_transform(input_data))

print('------------------------------------------------------------------------------------------')
print('X MSE: ', mean_squared_error(label_x, x_prediction))
print('Y MSE: ', mean_squared_error(label_y, y_prediction))

print('X r2 ', r2_score(label_x, x_prediction))
print('Y r2 ', r2_score(label_y, y_prediction))

results = {'PR_prediction_X':list(x_prediction), 'PR_prediction_Y': list(y_prediction)}
PR_results_df = pd.DataFrame(results, columns=['PR_prediction_X', 'PR_prediction_Y'])

PR_results_df.to_csv('PR_results.csv')