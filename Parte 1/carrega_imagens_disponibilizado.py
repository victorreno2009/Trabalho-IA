import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns

# Função para carregar e processar imagens
def carregar_imagens(pasta_raiz, dimensao):
    caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
    caminho_pessoas.pop(0)
    C = len(caminho_pessoas)  # Número de classes
    X, Y = [], []
    
    for i, pessoa in enumerate(caminho_pessoas):
        imagens_pessoa = os.listdir(pessoa)
        for imagem in imagens_pessoa:
            caminho_imagem = os.path.join(pessoa, imagem)
            imagem_original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            imagem_redimensionada = cv2.resize(imagem_original, (dimensao, dimensao))
            x = imagem_redimensionada.flatten()
            x = np.append(x, 1)  # Adiciona termo de bias
            y = -np.ones(C)
            y[i] = 1  # One-hot encoding
            X.append(x)
            Y.append(y)
    
    X = np.array(X).T / 255.0  # Normalização
    Y = np.array(Y).T
    return X, Y

# Funções auxiliares
def sign(u):
    return np.where(u >= 0, 1, -1)

def calcular_metricas(y_pred, y_true):
    # Calcula acurácia, sensibilidade e especificidade
    acertos = np.sum(np.all(y_pred == y_true, axis=0))
    acuracia = acertos / y_true.shape[1]
    
    # Sensibilidade e especificidade por classe
    C, N = y_true.shape
    sensibilidade = []
    especificidade = []
    for c in range(C):
        true_positives = np.sum((y_pred[c, :] == 1) & (y_true[c, :] == 1))
        false_negatives = np.sum((y_pred[c, :] == -1) & (y_true[c, :] == 1))
        true_negatives = np.sum((y_pred[c, :] == -1) & (y_true[c, :] == -1))
        false_positives = np.sum((y_pred[c, :] == 1) & (y_true[c, :] == -1))
        
        sens = true_positives / (true_positives + false_negatives + 1e-10)
        espec = true_negatives / (true_negatives + false_positives + 1e-10)
        
        sensibilidade.append(sens)
        especificidade.append(espec)
    
    return acuracia, np.mean(sensibilidade), np.mean(especificidade)

def matriz_confusao(y_pred, y_true):
    # Calcula a matriz de confusão por classe
    C = y_true.shape[0]
    conf_matrix = np.zeros((C, C))
    for i in range(y_true.shape[1]):
        true_class = np.argmax(y_true[:, i])
        pred_class = np.argmax(y_pred[:, i])
        conf_matrix[true_class, pred_class] += 1
    return conf_matrix

# Modelos
def perceptron(X, Y, lr=0.01, epochs=100):
    p, N = X.shape
    C, _ = Y.shape
    W = np.random.rand(C, p) * 0.01
    for epoch in range(epochs):
        for i in range(N):
            xi = X[:, i].reshape(-1, 1)
            yi = Y[:, i].reshape(-1, 1)
            u = np.dot(W, xi)
            y_pred = sign(u)
            W += lr * np.dot((yi - y_pred), xi.T)
    return W

# Monte Carlo
def monte_carlo(X, Y, modelo, rodadas=50, taxa_treino=0.8):
    resultados = []
    matrizes_confusao = []
    for r in range(rodadas):
        print(f"Rodada {r+1}/{rodadas}")
        indices = np.arange(X.shape[1])
        np.random.shuffle(indices)
        X, Y = X[:, indices], Y[:, indices]
        n_treino = int(taxa_treino * X.shape[1])
        X_treino, Y_treino = X[:, :n_treino], Y[:, :n_treino]
        X_teste, Y_teste = X[:, n_treino:], Y[:, n_treino:]
        
        W = modelo(X_treino, Y_treino)
        y_pred = sign(np.dot(W, X_teste))
        
        acuracia, sens, espec = calcular_metricas(y_pred, Y_teste)
        resultados.append((acuracia, sens, espec))
        
        conf_matrix = matriz_confusao(y_pred, Y_teste)
        matrizes_confusao.append(conf_matrix)
        
        print(f"Acurácia: {acuracia:.2f}, Sensibilidade: {sens:.2f}, Especificidade: {espec:.2f}")
    
    return resultados, matrizes_confusao

# Carregar e testar
dimensao = 50  # Ajustável
pasta_raiz = "RecFac"
X, Y = carregar_imagens(pasta_raiz, dimensao)
resultados, matrizes_confusao = monte_carlo(X, Y, perceptron)

# Análises finais
acuracias = [res[0] for res in resultados]
sensibilidades = [res[1] for res in resultados]
especificidades = [res[2] for res in resultados]

# Estatísticas
print("\nResultados finais:")
print(f"Acurácia média: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
print(f"Sensibilidade média: {np.mean(sensibilidades):.2f}")
print(f"Especificidade média: {np.mean(especificidades):.2f}")

# Gráficos
plt.figure(figsize=(12, 6))
plt.boxplot([acuracias, sensibilidades, especificidades], labels=['Acurácia', 'Sensibilidade', 'Especificidade'])
plt.title("Boxplot das Métricas de Desempenho")
plt.show()

# Matrizes de Confusão
melhor_idx = np.argmax(acuracias)
pior_idx = np.argmin(acuracias)

plt.figure(figsize=(10, 5))
sns.heatmap(matrizes_confusao[melhor_idx], annot=True, fmt=".2f", cmap="Blues")
plt.title("Matriz de Confusão (Melhor Acurácia)")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(matrizes_confusao[pior_idx], annot=True, fmt=".2f", cmap="Reds")
plt.title("Matriz de Confusão (Pior Acurácia)")
plt.show()

