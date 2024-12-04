import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("spiral.csv", delimiter=',')
c1, c2 = [-1, 1]
C = 2

#FUNÇÃO DE ATIVAÇÃO: DEGRAU BIPOLAR - FUNÇÃO SINAL - SIGN FUNCTION
def sign(u):
    return 1 if u>=0 else -1


def EQM(X,Y,w):
    p_1,N = X.shape
    eq = 0
    for t in range(N):
        x_t = X[:,t].reshape(p_1,1)
        u_t = w.T@x_t
        d_t = Y[t]
        eq += (d_t-u_t[0,0])**2
    return eq/(2*N)


X = data[:, :2]
Y = data[:, 2]

#Visualização dos dados:
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
    -np.ones((1,N)),
    X)
)

#Modelo ADALINE:
max_epoch = 10000
pr = 1e-5
lr = 0.00002


#Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
w = np.zeros((3,1)) # todos nulos
w = np.random.random_sample((3,1))-.5 # parâmetros aleatórios entre -0.5 e 0.5

#plot da reta que representa o modelo do perceptron simples em sua inicialização:
x_axis = np.linspace(-15,15)
x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
plt.plot(x_axis,x2,color='k')

#treinamento:
EQM1 = 1
EQM2 = 0
epochs = 0
hist = []
while epochs<max_epoch and abs(EQM1-EQM2)>pr:
    EQM1 = EQM(X,Y,w)
    hist.append(EQM1)
    for t in range(N):
        x_t = X[:,t].reshape(p+1,1)
        u_t = w.T@x_t
        d_t = Y[t]
        e_t = d_t - u_t
        w = w + lr*e_t*x_t
    epochs+=1
    EQM2 = EQM(X,Y,w)
    plt.pause(.01)
    x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    plt.plot(x_axis,x2,color='k',alpha=.2)
        
hist.append(EQM2)



#fim do treinamento
x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
line = plt.plot(x_axis,x2,color='green',linewidth=3)
plt.show()

plt.plot(hist,color='blue',linewidth=2,label="EQM x Epocas")
plt.legend()
plt.grid()
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.show()