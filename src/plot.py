#%%

#imports here
import pandas as pd
import matplotlib.pyplot as plt
import datetime, pytz
from sklearn.model_selection import train_test_split

#%%

#plot test train data sets
df = pd.read_csv("../data/processed/bitstampUSD.csv")
X = df.drop(["Weighted_Price"], axis=1)
y = df["Weighted_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=(2./3.), shuffle=False)

plt.figure(figsize=(15,5))
plt.plot(y_train)
plt.plot(y_val)
plt.plot(y_test)
plt.title("Bitcoin Weighted Price USD")
plt.ylabel("Weighted Price USD")
plt.legend(["Train Data", "Val Data", "Test Data"])
plt.show()

#%%

#plot truth vs predicted serial vs predicted parallel
y_true = pd.DataFrame(y_test.reset_index(drop=True))
keras_pred = pd.read_csv("../data/predictions/keras_y_pred.csv")
elephas_pred = pd.read_csv("../data/predictions/elephas_y_pred.csv")
gboost_mllib_pred = pd.read_csv("../data/predictions/gboost_mllib_y_pred.csv")
gboost_sklearn_pred = pd.read_csv("../data/predictions/gboost_sklearn_y_pred.csv")
rf_mllib_pred = pd.read_csv("../data/predictions/rf_mllib_y_pred.csv")
rf_sklearn_pred = pd.read_csv("../data/predictions/rf_sklearn_y_pred.csv")
lr_mllib_pred = pd.read_csv("../data/predictions/lr_mllib_y_pred.csv")
lr_sklearn_pred = pd.read_csv("../data/predictions/lr_sklearn_y_pred.csv")


plt.figure(figsize=(15,5))
plt.plot(y_true)
# plt.plot(keras_pred)
# plt.plot(elephas_pred)
# plt.legend(["True", "Keras Predictions", "Elephas Predictions"])
# plt.plot(gboost_sklearn_pred, zorder=2)
# plt.plot(gboost_mllib_pred, zorder=1)
# plt.legend(["True", "Hist Gradient Boosting Scikit-learn", "Hist Gradient Boosting MLlib"])
plt.plot(rf_sklearn_pred)
plt.plot(rf_mllib_pred)
plt.legend(["True", "Random Forest Scikit-learn", "Random Forest MLlib"])
# plt.plot(lr_sklearn_pred)
# plt.plot(lr_mllib_pred)
# plt.legend(["True", "Linear Regression Scikit-learn", "Linear Regression MLlib"])
plt.title("Predicted Price vs. Truth")
plt.ylabel("Weighted Price USD")
plt.show()

#%%

results_df = pd.read_csv("../data/predictions/results.csv")
results_df = results_df.set_index("Model")
print(results_df.head())

# plot error for each model
mae_results = results_df['MAE'].sort_values(ascending=False)
mae_results.plot(kind="bar")
plt.title("MAE Results")
plt.ylabel("MAE")
plt.xticks(rotation=30)
plt.show()

#plot fit time
fit_time = results_df['Fit Time'].sort_values(ascending=False)
fit_time.plot(kind="bar")
plt.title("Fit Time Results")
plt.ylabel("Fit Time (s)")
plt.xticks(rotation=30)
plt.show()

# %%
