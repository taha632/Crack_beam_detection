import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class CrackModel:
    def __init__(self, chemin_fichier):
        self.data = pd.read_excel(chemin_fichier)
        self.model = LinearRegression()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()

    def _split_data(self, test_size=0.2, random_state=42):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        return mse


chemin_fichier = r"C:\Users\Taha\Desktop\Projet_fissures\Base_de_donnée.xlsx"
crack_model_instance = CrackModel(chemin_fichier)
crack_model_instance.train_model()
mse_result = crack_model_instance.evaluate_model()
print(f"Mean Squared Error: {mse_result}")
