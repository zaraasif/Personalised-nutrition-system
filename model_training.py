import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping


data_path = "ABBREV.csv"
data = pd.read_csv(data_path)


data_cleaned = data.fillna(data.select_dtypes(include=[np.number]).mean())


for column in data.columns:
    if data[column].dtype == 'object':  
        data_cleaned[column] = data[column].fillna(data[column].mode()[0])


selected_columns = [
    'Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)',
    'Calcium_(mg)', 'Iron_(mg)', 'Vit_C_(mg)', 'Vit_A_IU',
    'Sodium_(mg)', 'Potassium_(mg)', 'Cholestrl_(mg)', 'Energ_Kcal'
]


missing_columns = [col for col in selected_columns if col not in data_cleaned.columns]
if missing_columns:
    raise ValueError(f"The following columns are missing in the dataset: {missing_columns}")

data_selected = data_cleaned[selected_columns]


features = data_selected.drop(columns=['Energ_Kcal'])
target = data_selected['Energ_Kcal']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def create_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  
    ])
    return model


input_dim = X_train_scaled.shape[1]
model = create_model(input_dim)


model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


epochs = 30
batch_size = 32
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)


model.save("nutrition_model.h5")
print("Model saved as 'nutrition_model.h5'")

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
print("Scaler saved as 'scaler.pkl'")


test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")


y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")


print("Sample Input Predictions:")
for i in range(5):
    sample_input = X_test_scaled[i].reshape(1, -1)
    sample_prediction = model.predict(sample_input)[0][0]
    print(f"Input: {X_test.iloc[i].values}, Prediction: {sample_prediction}, Actual: {y_test.iloc[i]}")
