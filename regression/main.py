############################# Parte 1 - Analise

import pandas as pd

# Caminhos dos arquivos
insurance_path = "./insurance.csv"
boston_path = "./boston.csv"

# Carregar os datasets
insurance_df = pd.read_csv(insurance_path)
boston_df = pd.read_csv(boston_path)

# Exibir informações básicas dos datasets
insurance_info = insurance_df.info()
boston_info = boston_df.info()

# Mostrar as primeiras linhas de cada dataset
insurance_head = insurance_df.head()
boston_head = boston_df.head()

print(insurance_info, insurance_head, boston_info, boston_head)

############################# Parte 2 - Separação 80 20

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Pré-processamento do dataset de seguros
insurance_df_encoded = insurance_df.copy()
label_encoders = {}

# Converter variáveis categóricas em numéricas
categorical_columns = ["sex", "smoker", "region"]
for col in categorical_columns:
    le = LabelEncoder()
    insurance_df_encoded[col] = le.fit_transform(insurance_df_encoded[col])
    label_encoders[col] = le  # Guardar os encoders para referência

# Separar features e target para seguros
X_insurance = insurance_df_encoded.drop(columns=["charges"])
y_insurance = insurance_df_encoded["charges"]

# Dividir em treino e teste (80/20)
X_train_ins, X_test_ins, y_train_ins, y_test_ins = train_test_split(X_insurance, y_insurance, test_size=0.2, random_state=42)

# Pré-processamento do dataset de Boston (não há variáveis categóricas)
X_boston = boston_df.drop(columns=["MEDV"])
y_boston = boston_df["MEDV"]

# Dividir em treino e teste (80/20)
X_train_bos, X_test_bos, y_train_bos, y_test_bos = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)

# Exibir o resultado da divisão
print(X_train_ins.shape, X_test_ins.shape, X_train_bos.shape, X_test_bos.shape)

############################# Parte 3 - Modelagem

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Lista de modelos
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Função para treinar e avaliar modelos
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}
    predictions = {}  # Dicionário para armazenar os valores reais e preditos
    
    for name, model in models.items():
        errors = {"RMSE": [], "MAE": [], "R2": []}
        last_y_pred = None  # Armazena a última previsão para gráficos
        
        for _ in range(30):  # 30 repetições
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            last_y_pred = y_pred  # Salva a última previsão
            
            # Calcular métricas
            errors["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            errors["MAE"].append(mean_absolute_error(y_test, y_pred))
            errors["R2"].append(r2_score(y_test, y_pred))
        
        # Armazena as métricas médias
        results[name] = {
            "RMSE": np.mean(errors["RMSE"]),
            "MAE": np.mean(errors["MAE"]),
            "R2": np.mean(errors["R2"])
        }
        
        # Armazena os valores reais e preditos da última repetição
        predictions[name] = {"y_real": y_test, "y_pred": last_y_pred}
    
    return results, predictions

# Treinar e avaliar modelos para os dois datasets
results_insurance, preds_insurance = train_and_evaluate(models, X_train_ins, y_train_ins, X_test_ins, y_test_ins)
results_boston, preds_boston = train_and_evaluate(models, X_train_bos, y_train_bos, X_test_bos, y_test_bos)

print("\nresults_insurance: ", results_insurance)
print("\nresults_boston", results_boston)
print("\npreds_insurance: ", preds_insurance)
print("\npreds_boston", preds_boston)

############################# Parte 4 - Estatística

import matplotlib.pyplot as plt
import seaborn as sns

# Função para plotar gráficos de dispersão
def plot_scatter(predictions, dataset_name):
    plt.figure(figsize=(15, 5))
    
    for i, (model, data) in enumerate(predictions.items()):
        plt.subplot(1, 3, i+1)
        sns.scatterplot(x=data["y_real"], y=data["y_pred"], alpha=0.6)
        plt.plot([min(data["y_real"]), max(data["y_real"])], 
                 [min(data["y_real"]), max(data["y_real"])], 
                 color="red", linestyle="--")  # Linha ideal
        plt.xlabel("Valores Reais")
        plt.ylabel("Valores Preditos")
        plt.title(f"{model} - {dataset_name}")
    
    plt.tight_layout()
    plt.show()

# Função para plotar erros residuais
def plot_residuals(predictions, dataset_name):
    plt.figure(figsize=(15, 5))
    
    for i, (model, data) in enumerate(predictions.items()):
        residuals = data["y_real"] - data["y_pred"]
        
        plt.subplot(1, 3, i+1)
        sns.histplot(residuals, kde=True, bins=30)
        plt.axvline(x=0, color="red", linestyle="--")
        plt.xlabel("Erro Residual")
        plt.ylabel("Frequência")
        plt.title(f"{model} - {dataset_name}")
    
    plt.tight_layout()
    plt.show()

# Plotar gráficos para o dataset de seguro
plot_scatter(preds_insurance, "Seguro de Saúde")
plot_residuals(preds_insurance, "Seguro de Saúde")

# Plotar gráficos para o dataset de imóveis
plot_scatter(preds_boston, "Imóveis Boston")
plot_residuals(preds_boston, "Imóveis Boston")