import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

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

# Plotando o grafo
# plt.figure(1)

# for e in E:
#     plt.plot(P[e,0], P[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

# plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Plotando os vertices

# plt.title("Grafo desconexo")
# plt.axis("equal")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)

# Plotando todas as componentes conexas
nc, idx = sp.csgraph.connected_components(sp.csr_matrix(A), directed=False)

# plt.figure(2)

# for i in range(0,nc):
# 	plt.subplot(2, 3, i+1)

# 	c = np.where(idx == i)[0]

# 	Pc = P[c,:]
# 	Ac = A[np.ix_(c,c)]
# 	Ec = np.array(np.where(Ac==1)).T

# 	for e in Ec:
# 		plt.plot(Pc[e,0], Pc[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

# 	plt.scatter(Pc[:,0], Pc[:,1], c="darkgrey", s=1) # Plotando os vertices

# 	plt.title(f"Componente conexa {i+1}")
# 	plt.axis("equal")
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.grid(True)

# Obtendo maior componente conexa
bc = np.where(idx == np.argmax(np.bincount(idx)))[0] # maior componente conexa

P = P[bc,:]
A = A[np.ix_(bc,bc)]
E = np.array(np.where(A==1)).T

# Plotando maior componente conexa
# plt.figure(3)

# for e in E:
#     plt.plot(P[e,0], P[e,1], color="lightsteelblue", linewidth=0.5) # Plotando arestas

# plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Plotando os vertices

# plt.title("Maior componente conexa")
# plt.axis("equal")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)

# Construindo o sistema
G = np.diag(A.sum(axis=1)) # Matriz de grau
L = G-A # Matriz Laplaciana

n = A.shape[0]
k = int(0.2/100 * n)

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

f = plt.figure(5)

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