import numpy as np
import matplotlib.pyplot as plt

# Carregamento dos dados
data = np.loadtxt("spiral.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2].reshape(-1, 1)

# Visualização dos dados
plt.scatter(X[Y.flatten() == -1, 0], X[Y.flatten() == -1, 1], color='red', edgecolor='k', alpha=0.5, label="Classe -1")
plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], color='blue', edgecolor='k', alpha=0.5, label="Classe 1")
plt.legend()
plt.show()

# Normalização dos dados
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Configuração do MLP
np.random.seed(42)

input_size = 2  # Duas características no dataset
hidden_size = 10  # Número de neurônios na camada oculta
output_size = 1  # Saída binária (1 ou -1)
lr = 0.01  # Taxa de aprendizado
epochs = 1000  # Número de épocas

# Inicialização dos pesos e bias
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Funções de ativação
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Treinamento do MLP
losses = []
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Cálculo do erro (loss)
    loss = np.mean((A2 - Y) ** 2)
    losses.append(loss)
    
    # Backward pass
    dA2 = 2 * (A2 - Y) / Y.size
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Atualização dos pesos
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    # Exibição do progresso
    if epoch % 100 == 0:
        print(f"Época {epoch}, Perdas: {loss:.4f}")

# Visualização da loss
plt.plot(losses)
plt.title("Perdas durante o treinamento")
plt.xlabel("Épocas")
plt.ylabel("Perdas")
plt.show()

# Visualização da decisão
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z1 = np.dot(grid, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)
preds = (A2 > 0.5).astype(int) * 2 - 1  # Converte para -1 ou 1
preds = preds.reshape(xx.shape)

plt.contourf(xx, yy, preds, alpha=0.5, cmap='coolwarm')
plt.scatter(X[Y.flatten() == -1, 0], X[Y.flatten() == -1, 1], color='red', edgecolor='k')
plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], color='blue', edgecolor='k')
plt.title("Fronteira de decisão do MLP")
plt.show()
