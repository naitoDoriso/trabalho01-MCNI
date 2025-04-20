import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

# Coordenadas
P = []

f = open("manh.xy", 'r')
for line in f:
	if line.strip():
		P.append(list(map(float, line.strip().split())))
P = np.array(P)
f.close()

# Matriz de adjacencia
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

# Plotando o grafo desconexo
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

# Plotando todas as componentes conexas
nc, idx = sp.csgraph.connected_components(sp.csr_matrix(A), directed=False)

f = plt.figure(2,figsize=(15,12))

for i in range(0,nc):
	plt.subplot(2, 3, i+1)

	c = np.where(idx == i)[0]

	Pc = P[c,:]
	Ac = A[np.ix_(c,c)]
	Ec = np.array(np.where(Ac==1)).T

	for e in Ec:
		plt.plot(Pc[e,0], Pc[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

	plt.scatter(Pc[:,0], Pc[:,1], c="darkgrey", s=1) # Plotando os vertices

	plt.title(f"Componente conexa {i+1}")
	plt.axis("equal")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)

f.savefig("fig2.pdf")

# Obtendo maior componente conexa
bc = np.where(idx == np.argmax(np.bincount(idx)))[0] # maior componente conexa

P = P[bc,:]
A = A[np.ix_(bc,bc)]
E = np.array(np.where(A==1)).T

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

# Construindo o sistema
G = np.diag(A.sum(axis=1)) # Matriz de grau
L = G-A # Matriz Laplaciana

n = A.shape[0]
k = int(5/100 * n) # 5% dos pontos

np.random.seed(42)
idx = np.random.choice(np.arange(0, n), size=k, replace=False) # Escolhendo 5% de pontos para condicao de contorno
b = np.zeros((n,1))
b[idx,0] = np.random.uniform(20, 30, size=k) # Escolhendo temperaturas T, com 20°C <= T < 30°C para condicao de contorno

alpha = 1.0e7
MP = np.zeros((n,n)) # Matriz de penalidade
MP[idx,idx] = alpha

# Montando o sistema M*x = c
M = L+MP
c = MP@b

# Plotando a maior componente conexa com os pontos fixos
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

# Resolvendo o sistema de forma direta
T = np.linalg.solve(M,c)

def funcao_cholesky(M, c):
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


def funcao_lu(M,c):
	#Calculando a demposicao LU: M= P*L*U, onde P eh uma matriz de permutacao
	P,L,U = la.lu(M)

	#Resolvendo Ly=Pc
	Pc= P @ c
	y = la.solve_triangular(L, Pc, lower=True)

	#Resolve Ux=y
	x = la.solve_triangular(U, y, lower=False)

	return x

def funcao_qr(M, c):

	#Calculando a decomp. QR: M=QR
	Q, R = la.qr(M)

	#Resolvendo Rx=Q.Tc
	Qtc = Q.T @ c
	x = la.solve_triangular(R, Qtc, lower=False)
	return x

import time

tempos = []
metods = ["np.linalg.solve", "Cholesky", "LU", "QR"]

#Tempo metodo direto
beg= time.time()
T_numpy = np.linalg.solve(M,c)
end = time.time()
tempos.append(end-beg)
print(f"Tempo de execução usando numpy.linalg.solve: {tempos[0]:.6f} segundos")

#Tempo Metodo Cholesky

beg1= time.time()
T_cholesky = funcao_cholesky(M,c)
end1 = time.time()
tempos.append(end1-beg1)
print(f"Tempo de execução usando Cholesky: {tempos[1]:.6f} segundos")

#Tempo Metodo LU
beg2 = time.time()
T_lu = funcao_lu(M,c)
end2 = time.time()
tempos.append(end2-beg2)
print(f"Tempo de execução usando LU: {tempos[2]:.6f} segundos")

#Tempo Metodo QR
beg3 = time.time()
T_qr = funcao_qr(M, c)
end3 = time.time()
tempos.append(end3 - beg3)
print(f"Tempo de execução usando QR: {tempos[3]:.6f} segundos")

# Verificando se todas as soluções são equivalentes
print("\nVerificação de equivalência das soluções:")
print(f"np.linalg.solve vs Cholesky: {np.allclose(T_numpy, T_cholesky)}")
print(f"np.linalg.solve vs LU: {np.allclose(T_numpy, T_lu)}")
print(f"np.linalg.solve vs QR: {np.allclose(T_numpy, T_qr)}")

# Plotando o graf de barras comparando os tempos
f = plt.figure(6, figsize=(10, 6))
plt.bar(metods, tempos, color='skyblue')
plt.ylabel('Tempo de execução (segundos)')
plt.title('Comparação de tempo entre diferentes métodos de solução')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adicionando o tempo em cima das barras
for i, tempo in enumerate(tempos):
    plt.text(i, tempo + max(tempos)*0.01, f'{tempo:.6f}',
             ha='center', va='bottom', fontsize=9)

f.savefig("fig6.pdf")

T = T_cholesky

f = plt.figure(5)

T = funcao_cholesky(M,c)
for e in E:
    plt.plot(P[e,0], P[e,1], color="gainsboro", linewidth=0.5) # Plotando arestas

plt.scatter(P[:,0], P[:,1], c=T, cmap="jet", s=1, zorder=2) # Plotando os vertices
plt.colorbar()

plt.title(r"Temperatura por resolução do sistema linear diretamente")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

f.savefig("fig5.pdf")

plt.show()
