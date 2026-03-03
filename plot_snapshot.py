import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"out/snapshot_0001000.npz")  # change if needed
u = data["u"]
v = data["v"]
Lx = float(data["Lx"])
Ly = float(data["Ly"])

nx, ny = u.shape
x = np.linspace(0, Lx, nx, endpoint=False) + 0.5*Lx/nx
y = np.linspace(0, Ly, ny, endpoint=False) + 0.5*Ly/ny
X, Y = np.meshgrid(x, y, indexing="ij")

speed = np.sqrt(u*u + v*v)

plt.figure()
plt.contourf(X, Y, speed, levels=40)
plt.colorbar(label="|u|")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Speed magnitude")
plt.tight_layout()
plt.show()

plt.figure()
skip = (slice(None, None, 6), slice(None, None, 6))
plt.quiver(X[skip], Y[skip], u[skip], v[skip])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Velocity vectors")
plt.tight_layout()
plt.show()
