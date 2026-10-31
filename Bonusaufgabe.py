import numpy as np
import matplotlib.pyplot as plt

# Random Seed festlegen, damit die Ergebnisse reproduzierbar sind
np.random.seed(42)


"""
lineare Klassifikation
"""

coords_x1 = np.random.normal(0, 0.5, 100)
coords_x2 = np.random.normal(0, 0.5, 100)
dx, dy = np.random.uniform(-0.4, 0.4, 2)
class_y = (((coords_x1 - dx)**2 + (coords_x2 - dy)**2) < 0.4)

# class y von True/False in 1 und -1 umwandeln
mapped_y = class_y.astype(int) * 2 - 1


c = list(map(lambda x: 'r' if x == 1 else 'b', mapped_y))
plt.scatter(coords_x1, coords_x2, c=c)
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()

# Design Matrix X erstellen, die die Features für die lineare Regression enthält
X = np.stack([np.ones_like(coords_x1), coords_x1, coords_x2, coords_x1**2 , coords_x2**2 , coords_x1*coords_x2 ], axis= 1)

# Omega-Gewichte berechnen
w = np.linalg.inv(X.T @ X) @ X.T @ mapped_y
print(X)
print(w)

# Rauschen zum Gewicht hinzufügen
w += np.random.uniform(-1.5 , 1.5, w.shape)

# Vorhersage der Klasse für die Trainingsdaten
pred_y = X @ w
print(pred_y)


# Plotten der Ergebnisse
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(coords_x1, coords_x2, pred_y, c=c)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Klasse")
plt.show()

x1_grid, x2_grid = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
z = w[0] + w[1]*x1_grid + w[2]*x2_grid + w[3]*x1_grid**2 + w[4]*x2_grid**2 + w[5]*x1_grid*x2_grid
c_pred = list(map(lambda x: 'b' if x < 0 else 'r', pred_y))

plt.contourf(x1_grid, x2_grid, z, levels=[-20, 0, 20], colors=["b", "r"], alpha=.2)
plt.contour(x1_grid, x2_grid, z, levels=[0], colors=["k"])
plt.scatter(coords_x1, coords_x2, c=c_pred)
plt.show()


"""
logistische Klassifikation
"""

y = class_y.astype(int)

# Klassenzugehörigkeitswahrscheinlichkeiten
q = 1 / (1 + np.exp(-pred_y))

print((q > 0.5) == y)

# ! erstmal testen
L = np.prod((q ** y) * (1 - q)**(1-y))

print(L)

# Log-Likelihood mit der Summe über die Logarithmen
lnLcheck = np.sum((np.log((q ** y) * (1 - q)**(1-y))))
print(lnLcheck)

# direkts ln auf die likelihood
log_L = np.log(L)
print(log_L)

lernrate = 0.005

def log_likelihood_gradienten_berechnen(X, w):

    pred_y = X @ w
    q = 1 / (1 + np.exp(-pred_y))

    gradient = X.T @ (y - q)

    return w + lernrate * gradient

plt.ion()

for i in range(0, 1000):

    # jeden Durchgang den Gradientenaufstieg berechnen
    w = log_likelihood_gradienten_berechnen(X, w)
    print(f"{i+1}. {w}")

    # Iteration plotten
    pred_y = X @ w

    z = w[0] + w[1]*x1_grid + w[2]*x2_grid + w[3]*x1_grid**2 + w[4]*x2_grid**2 + w[5]*x1_grid*x2_grid
    c_pred = list(map(lambda x: 'b' if x < 0 else 'r', pred_y))

    plt.clf()
    plt.contourf(x1_grid, x2_grid, z, levels=[-20, 0, 20], colors=["b", "r"], alpha=.2)
    plt.contour(x1_grid, x2_grid, z, levels=[0], colors=["k"])
    plt.scatter(coords_x1, coords_x2, c=c_pred)
    
    plt.pause(0.01)

plt.ioff()
plt.show()