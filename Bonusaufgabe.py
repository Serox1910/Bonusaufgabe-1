import numpy as np
import matplotlib.pyplot as plt

coords_x1 = np.random.normal(0, 0.5, 100)
coords_x2 = np.random.normal(0, 0.5, 100)
dx, dy = np.random.uniform(-0.4, 0.4, 2)
class_y = (((coords_x1 - dx)**2 + (coords_x2 - dy)**2) < 0.4)

class_y = np.where(class_y, 1, -1)



c = list(map(lambda x: 'r' if x == 1 else 'b', class_y))
plt.scatter(coords_x1, coords_x2, c=c)
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()

X = np.stack([np.ones_like(coords_x1), coords_x1, coords_x2, coords_x1**2 , coords_x2**2 , coords_x1*coords_x2 ], axis= 1)
w = np.linalg.inv(X.T @ X) @ X.T @ class_y
print(X)
print(w)

w += np.random.uniform(-1.5 , 1.5, w.shape)
pred_y = X @ w
print(pred_y)

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


