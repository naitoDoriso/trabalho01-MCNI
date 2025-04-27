import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import time
import os

# Funcoes dos metodos diretos
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

# Funcoes dos metodos iterativos
def GaussJacobi(M,c):
	n = M.shape[0]
	D = np.diag(np.diag(M))
	C = np.eye(n)-np.linalg.solve(D,M)
	g = np.linalg.solve(D,c)

	erro = 0
	k = 0
	kmax = 1e5
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
	kmax = 1e5
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

	a = (r0.T @ r0)/(p0.T @ M @ p0)
	x = x0 + a*p0
	r = r0 - a*(M @ p0)
	p = p0

	erro = 0
	k = 1
	kmax = 1e5
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

f = open("manh.el", 'r')
for line in f:
	if line.strip():
		i, j = map(int, line.strip().split())
		A[i,j] = 1
		A[j,i] = 1
f.close()

# Plotando o grafo desconexo
f = plt.figure(1)

E = np.array(np.where(np.triu(A) == 1)).T
for e in E:
    plt.plot(P[e,0], P[e,1], color="lightsteelblue", linewidth=0.5) # Arestas

plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Vertices

plt.title("Grafo desconexo")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

if not os.path.exists("figs"):
    os.makedirs("figs")

f.savefig("figs/fig1.pdf")

# Plotando todas as componentes conexas
nc, idx = sp.csgraph.connected_components(sp.csr_matrix(A), directed=False)

f = plt.figure(2,figsize=(23,14))

for i in range(0,nc):
	plt.subplot(2, 3, i+1)

	c = np.where(idx == i)[0]

	Pc = P[c,:]
	Ac = A[np.ix_(c,c)]
	Ec = np.array(np.where(np.triu(Ac)==1)).T

	for e in Ec:
		plt.plot(Pc[e,0], Pc[e,1], color="lightsteelblue", linewidth=0.5) # Arestas

	plt.scatter(Pc[:,0], Pc[:,1], c="darkgrey", s=1) # Vertices

	plt.title(f"Componente conexa {i+1}")
	plt.axis("equal")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)
 
f.savefig("figs/fig2.pdf")

bc = np.where(idx == np.argmax(np.bincount(idx)))[0] # Maior componente conexa

P = P[bc,:]
A = A[np.ix_(bc,bc)]
E = np.array(np.where(np.triu(A)==1)).T

# Plotando maior componente conexa
f = plt.figure(3)

for e in E:
    plt.plot(P[e,0], P[e,1], color="lightsteelblue", linewidth=0.5) # Arestas

plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Vertices

plt.title("Maior componente conexa")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

f.savefig("figs/fig3.pdf")

def solve(i0, b, ix):
	# Condicao de contorno
	n = A.shape[0]
	k = ix.shape[0]

	# Plotando a maior componente conexa com os pontos fixos
	f = plt.figure(i0,figsize=(10, 6))

	for e in E:
		plt.plot(P[e,0], P[e,1], color="gainsboro", linewidth=0.5) # Arestas

	plt.scatter(P[:,0], P[:,1], c="darkgrey", s=1) # Vertices
	plt.scatter(P[ix,0], P[ix,1], c=b[ix], cmap="jet", s=2, zorder=2) # Vertices fixos
	plt.colorbar()

	plt.title(f"Temperatura fixa em {k} pontos de um total de {n}")
	plt.axis("equal")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)

	f.savefig(f"figs/fig{i0}.pdf")

	# Montando o sistema
	G = np.diag(A.sum(axis=1)) # Matriz de grau
	L = G-A # Matriz Laplaciana

	alpha = 1.0e7
	MP = np.zeros((n,n)) # Matriz de penalidade
	MP[ix,ix] = alpha

	M = L+MP # M = (L+P)
	c = MP@b # c = Pb
	# (L+P)T = Pb <=> M * T = c

	# Resolvendo o sistema por metodos diretos
	tempos = []
	metods = ["np.linalg.solve", "Cholesky", "LU", "QR"]

	beg= time.time()
	Td = np.linalg.solve(M,c)
	end = time.time()
	tempos.append(end-beg)

	beg1= time.time()
	Tc = Cholesky(M,c)
	end1 = time.time()
	tempos.append(end1-beg1)

	beg2 = time.time()
	Tl = LU(M,c)
	end2 = time.time()
	tempos.append(end2-beg2)

	beg3 = time.time()
	Tq = QR(M, c)
	end3 = time.time()
	tempos.append(end3 - beg3)

	# Plotando o grafico de barras, comparando os tempos dos metodos diretos
	f = plt.figure(i0+1, figsize=(10, 6))

	plt.bar(metods, tempos, color='skyblue')
	plt.ylabel('Tempo de execução (segundos)')
	plt.title('Comparação de tempo entre métodos diretos')
	plt.grid(True, axis='y', linestyle='--', alpha=0.7)

	# Adicionando o tempo em cima das barras
	for i, tempo in enumerate(tempos):
		plt.text(i, tempo + max(tempos)*0.01, f'{tempo:.6f}',
				ha='center', va='bottom', fontsize=9)

	f.savefig(f"figs/fig{i0+1}.pdf")

	# Resolvendo o sistema por metodos iterativos
	passos = []
	tempos = []
	metods = ["Gauss Jacobi", "Gauss Seidel", "Gradientes Conjugados"]

	st = time.time()
	[Tj,k,_] = GaussJacobi(M,c)
	end = time.time()
	passos.append(k)
	tempos.append(end-st)

	st = time.time()
	[Ts,k,_] = GaussSeidel(M,c)
	end = time.time()
	passos.append(k)
	tempos.append(end-st)

	st = time.time()
	[Tg,k,_] = GradConj(M,c)
	end = time.time()
	passos.append(k)
	tempos.append(end-st)

	# Plotando o grafico de barras, comparando os tempos e passos dos metodos iterativos
	f = plt.figure(i0+2, figsize=(10, 6))

	plt.bar(metods, tempos, color='skyblue')
	plt.ylabel('Tempo de execução (segundos)')
	plt.title('Comparação de tempo entre métodos iterativos')
	plt.grid(True, axis='y', linestyle='--', alpha=0.7)

	# Adicionando o tempo em cima das barras
	for i, tempo in enumerate(tempos):
		plt.text(i, tempo + max(tempos)*0.01, f'{tempo:.6f}',
				ha='center', va='bottom', fontsize=9)
	
	print(passos) # Mostra a quantidade de passos necessarios para convergencia dos 3 metodos

	f.savefig(f"figs/fig{i0+2}.pdf")

	# Plotando a interpolacao no grafico
	T = Tc # Cholesky
	f = plt.figure(i0+3)

	for e in E:
		plt.plot(P[e,0], P[e,1], color="gainsboro", linewidth=0.5) # Arestas

	plt.scatter(P[:,0], P[:,1], c=T, cmap="jet", s=1, zorder=2) # Vertices
	plt.colorbar()

	plt.title("Temperatura interpolada")
	plt.axis("equal")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)

	f.savefig(f"figs/fig{i0+3}.pdf")

	# Analisando o erro dos metodos diretos
	f = plt.figure(i0+4, figsize=(10, 6))

	r = np.array([np.linalg.norm(c-M@Td), np.linalg.norm(c-M@Tc), np.linalg.norm(c-M@Tl), np.linalg.norm(c-M@Tq)])
	plt.bar(["np.linalg.solve", "Cholesky", "LU", "QR"], r, color='skyblue')
	plt.ylabel('Erro')
	plt.yscale('log', base=10)
	plt.title('Comparação do resíduo entre métodos diretos')
	plt.grid(True, axis='y', linestyle='--', alpha=0.7)

	# Adicionando o erro em cima das barras
	for i, x in enumerate(r):
		plt.text(i, x + max(r)*0.01, f'{x:.7f}',
				ha='center', va='bottom', fontsize=9)

	f.savefig(f"figs/fig{i0+4}.pdf")

	# Analisando o erro dos metodos iterativos
	f = plt.figure(i0+5, figsize=(10, 6))

	r = np.array([np.linalg.norm(c-M@Tj), np.linalg.norm(c-M@Ts), np.linalg.norm(c-M@Tg)])
	plt.bar(["Gauss Jacobi", "Gauss Seidel", "Gradientes Conjugados"], r, color='skyblue')
	plt.ylabel('Erro')
	plt.yscale('log', base=10)
	plt.title('Comparação do resíduo entre métodos iterativos')
	plt.grid(True, axis='y', linestyle='--', alpha=0.7)

	# Adicionando o erro em cima das barras
	for i, x in enumerate(r):
		plt.text(i, x + max(r)*0.01, f'{x:.6f}',
				ha='center', va='bottom', fontsize=9)

	f.savefig(f"figs/fig{i0+5}.pdf")

# Escolhendo 9 pontos fixos manualmente
n = A.shape[0]
P_ord = np.argsort(P[:,1])[::-1] # Ordena em ordem decrescente de acordo com y

aux = np.linspace(0,n-4,9,dtype=int)
ix = P_ord[aux] # Pega 9 pontos, em ordem decrescente de acordo com o y
b = np.zeros((n,1))
b[ix,0] = np.linspace(20,24,9) # Atribui as menores temperaturas para os mais altos e, as maiores para os mais baixos

tol = 1.0e-1
solve(4, b, ix) # Figura 4 ate Figura 9

# Escolhendo 5% de pontos fixos aleatoriamente
np.random.seed(42)
k = int(5/100 * n)

ix = np.random.choice(np.arange(0, n), size=k, replace=False) # Escolhendo 5% de pontos para condicao de contorno
b = np.zeros((n,1))
b[ix,0] = np.random.uniform(20, 24, size=k) # Escolhendo temperaturas T, com 20°C <= T < 24°C para condicao de contorno

tol = 1.0e-3
solve(10, b, ix) # Figura 10 ate Figura 15