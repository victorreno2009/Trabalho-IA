import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("spiral.csv", delimiter=',')
c1, c2 = [-1, 1]
C = 2

#FUNÇÃO DE ATIVAÇÃO: DEGRAU BIPOLAR - FUNÇÃO SINAL - SIGN FUNCTION
def sign(u):
    return 1 if u>=0 else -1

X = data[:, :2]
Y = data[:, 2]

#Visualização dos dados:
dd = data[:,2]
plt.scatter(X[Y==c1,0],X[Y==c1,1],color='red',edgecolor='k', alpha=.5)
plt.scatter(X[Y==c2,0],X[Y==c2,1],color='blue',edgecolor='k', alpha=.5)
plt.legend()
plt.ylim(np.min(X), np.max(X))
plt.xlim(np.min(X), np.max(X))

#Organização dos dados:
#Passo 1: Organizar os dados de treinamento com a dimensão (p x N)
X = X.T
Y = Y.T
p,N = X.shape

#Passo 2: Adicionar o viés (bias) em cada uma das amostras:
X = np.concatenate((
    np.ones((1,N)),
    X)
)

#Modelo do Perceptron Simples:
lr = .001 # Definição do hiperparâmetro Taxa de Aprendizado (Learning Rate)

#Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
w = np.zeros((3,1)) # todos nulos
w = np.random.random_sample((3,1))-.5 # parâmetros aleatórios entre -0.5 e 0.5

#plot da reta que representa o modelo do perceptron simples em sua inicialização:
x_axis = np.linspace(-15,15)
x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
plt.plot(x_axis,x2,color='blue')



#condição inicial
erro = True
epoca = 0 #inicialização do contador de épocas.
while(erro):
    erro = False
    for t in range(N):
        x_t = X[:,t].reshape(p+1,1)
        u_t = (w.T@x_t)[0,0]
        y_t = sign(u_t)
        d_t = float(Y[t])
        e_t = d_t - y_t
        w = w + (lr*e_t*x_t)/2
        if(y_t!=d_t):
            erro = True
    #plot da reta após o final de cada época
    plt.pause(.0001)    
    x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    plt.plot(x_axis,x2,color='orange',alpha=.1)
    epoca+=1

#fim do treinamento
x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
line = plt.plot(x_axis,x2,color='green',linewidth=3)
plt.show()