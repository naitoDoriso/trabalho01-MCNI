import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import time

## Coordenadas
P = []

f = open("manh.xy", 'r')
for line in f:
	if line.strip():
		P.append(list(map(float, line.strip().split())))
P = np.array(P)
f.close()

## Matriz de adjacencia
n = P.shape[0]

A = np.zeros((n,n))
E = []

f = open("manh.el", 'r')
for line in f:
	if line.strip():
		i, j = map(int, line.strip().split())
		E.append([i,j])
		A[i,j] = 1
		A[j,i] = 1
E = np.array(E)
f.close()

## Plotando o grafo desconexo
f = plt.figure(1)

for e in E:
    plt.plot(P[e,0], P[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Plotando os vertices

plt.title("Grafo desconexo")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

f.savefig("fig1.pdf")

## Plotando todas as componentes conexas
nc, idx = sp.csgraph.connected_components(sp.csr_matrix(A), directed=False)

f = plt.figure(2,figsize=(15,12))

for i in range(0,nc):
	plt.subplot(2, 3, i+1)

	c = np.where(idx == i)[0]

	Pc = P[c,:]
	Ac = A[np.ix_(c,c)]
	Ec = np.array(np.where(np.triu(Ac)==1)).T

	for e in Ec:
		plt.plot(Pc[e,0], Pc[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

	plt.scatter(Pc[:,0], Pc[:,1], c="darkgrey", s=1) # Plotando os vertices

	plt.title(f"Componente conexa {i+1}")
	plt.axis("equal")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)
 
f.savefig("fig2.pdf")

## Maior componente conexa
bc = np.where(idx == np.argmax(np.bincount(idx)))[0] # maior componente conexa

P = P[bc,:]
A = A[np.ix_(bc,bc)]
E = np.array(np.where(np.triu(A)==1)).T

# Plotando maior componente conexa
f = plt.figure(3)

for e in E:
    plt.plot(P[e,0], P[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Plotando os vertices

plt.title("Maior componente conexa")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

f.savefig("fig3.pdf")

## Condicao de contorno
n = A.shape[0]
k = int(5/100 * n) # 5% dos pontos

np.random.seed(42)
idx = np.random.choice(np.arange(0, n), size=k, replace=False) # Escolhendo 5% de pontos para condicao de contorno
b = np.zeros((n,1))
b[idx,0] = np.random.uniform(20, 30, size=k) # Escolhendo temperaturas T, com 20°C <= T < 30°C para condicao de contorno

## Plotando a maior componente conexa com os pontos fixos
f = plt.figure(4)

for e in E:
    plt.plot(P[e,0], P[e,1], color="gainsboro", linewidth=0.5) # Plotando arestas

plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Plotando os vertices
plt.scatter(P[idx,0], P[idx,1], c=b[idx], cmap="jet", s=2, zorder=2) # Plotando os vertices fixos
plt.colorbar()

plt.title(r"Temperatura fixa em 5% dos pontos")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

f.savefig("fig4.pdf")

## Montando o sistema M*x = c
G = np.diag(A.sum(axis=1)) # Matriz de grau
L = G-A # Matriz Laplaciana

alpha = 1.0e7
MP = np.zeros((n,n)) # Matriz de penalidade
MP[idx,idx] = alpha

M = L+MP
c = MP@b

## Resolvendo o sistema por metodos diretos
T = np.linalg.solve(M,c)

def Cholesky(M, c):
    """
    Resolve o sistema M*x = c usando a decomposição de Cholesky.

    Parametros:
    M (numpy.ndarray): Matriz M>0 positiva
    c (numpy.ndarray): Vetor resultante

    Retorna:
    numpy.ndarray: O vetor sol. x onde M*x = c
    """
    # Calcula a decomp. de Cholesky: M = L*L^T
    L = la.cholesky(M, lower=True)

    # Resolve a matriz triangular L*y = c
    y = la.solve_triangular(L, c, lower=True)

    # Resolve a transposta da matriz triang L^T*x = y
    x = la.solve_triangular(L.T, y, lower=False)

    return x


def LU(M,c):
	#Calculando a demposicao LU: M= P*L*U, onde P eh uma matriz de permutacao
	P,L,U = la.lu(M)

	#Resolvendo Ly=Pc
	Pc= P @ c
	y = la.solve_triangular(L, Pc, lower=True)

	#Resolve Ux=y
	x = la.solve_triangular(U, y, lower=False)

	return x

def QR(M, c):

	#Calculando a decomp. QR: M=QR
	Q, R = la.qr(M)

	#Resolvendo Rx=Q.Tc
	Qtc = Q.T @ c
	x = la.solve_triangular(R, Qtc, lower=False)
	return x

tempos = []
metods = ["np.linalg.solve", "Cholesky", "LU", "QR"]

#Tempo Metodo Direto
beg= time.time()
aux = np.linalg.solve(M,c)
end = time.time()
tempos.append(end-beg)

#Tempo Metodo Cholesky
beg1= time.time()
T = Cholesky(M,c)
end1 = time.time()
tempos.append(end1-beg1)

#Tempo Metodo LU
beg2 = time.time()
LU(M,c)
end2 = time.time()
tempos.append(end2-beg2)

#Tempo Metodo QR
beg3 = time.time()
QR(M, c)
end3 = time.time()
tempos.append(end3 - beg3)

## Plotando o grafico de barras comparando os tempos
f = plt.figure(5, figsize=(10, 6))

plt.bar(metods, tempos, color='skyblue')
plt.ylabel('Tempo de execução (segundos)')
plt.title('Comparação de tempo entre diferentes métodos de solução')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adicionando o tempo em cima das barras
for i, tempo in enumerate(tempos):
    plt.text(i, tempo + max(tempos)*0.01, f'{tempo:.6f}',
             ha='center', va='bottom', fontsize=9)

f.savefig("fig5.pdf")

## Resolvendo o sistema por metodos iterativos
def GaussJacobi(M,c):
	n = M.shape[0]
	D = np.diag(np.diag(M))
	C = np.eye(n)-np.linalg.solve(D,M)
	g = np.linalg.solve(D,c)

	erro = 0
	k = 0
	kmax = 1e4
	x0 = np.zeros((n,1))
	while (np.linalg.norm(c-M@x0) > tol and k < kmax):
		k = k+1
		x0 = C@x0 + g
	if (k == kmax):
		erro = 1
	return [x0,k,erro]

def GaussSeidel(M,c):
	n = M.shape[0]
	L = np.tril(M)
	R = np.triu(M,1)
	C = -np.linalg.solve(L,R)
	g = np.linalg.solve(L,c)

	erro = 0
	k = 0
	kmax = 1e4
	x0 = np.zeros((n,1))
	while (np.linalg.norm(c-M@x0) > tol and k < kmax):
		k = k+1
		x0 = C@x0 + g
	if (k == kmax):
		erro = 1
	return [x0,k,erro]

def GradConj(M,c):
	n = M.shape[0]
	x0 = np.zeros((n,1))
	r0 = c-(M @ x0)
	p0 = r0

	a = (r0.T @ r0)/(r0.T @ M @ r0)
	x = x0 + a*p0
	r = r0 - a*(M @ p0)
	p = p0

	erro = 0
	k = 1
	kmax = 1e4
	while (np.linalg.norm(r) > tol and k < kmax):
		x0 = x
		p0 = p

		b = (r.T @ r)/(r0.T @ r0)
		p = r + b*p0
		r0 = r

		a = (r.T @ r)/(p.T @ M @ p)
		x = x0 + a*p
		r = r0 - a*(M @ p)

		k = k+1
	if (k == kmax):
		erro = 1
	return [x,k,erro]

## Calcundo o numero de iteracoes e os tempos
tol = 1.0e-3

st = time.time()
GaussJacobi(M,c)
end = time.time()
TJ = end-st

st = time.time()
GaussSeidel(M,c)
end = time.time()
TS = end-st

st = time.time()
GradConj(M,c)
end = time.time()
TG = end-st

f = plt.figure(6)

for e in E:
    plt.plot(P[e,0], P[e,1], color="gainsboro", linewidth=0.5) # Plotando arestas

plt.scatter(P[:,0], P[:,1], c=T, cmap="jet", s=1, zorder=2) # Plotando os vertices
plt.colorbar()

plt.title(r"Temperatura interpolada")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

f.savefig("fig6.pdf")

f = plt.figure(7, figsize=(10, 6))

tempos = [TJ,TS,TG]
plt.bar(["Gauss Jacobi", "Gauss Seidel", "Gradientes Conjugados"], tempos, color='skyblue')
plt.ylabel('Tempo de execução (segundos)')
plt.title('Comparação de tempo entre diferentes métodos de solução')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adicionando o tempo em cima das barras
for i, tempo in enumerate(tempos):
    plt.text(i, tempo + max(tempos)*0.01, f'{tempo:.6f}',
             ha='center', va='bottom', fontsize=9)

f.savefig("fig7.pdf")

plt.show()
