import numpy as np
import matplotlib.pyplot as plt


def linear_congruential_generator(seed, a, c, m, n):
    random_numbers = np.zeros(n)
    random_numbers[0] = seed

    for i in range(1, n):
        random_numbers[i] = (a * random_numbers[i - 1] + c) % m

    return random_numbers / m

def random_walk_3d(xi, yi, zi, n, Rmove):
    x, y, z = xi, yi, zi
    xf, yf, zf = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        ri = Rmove[i]
        move = int(ri * 6) - 3  # Movimiento en 3 dimensiones, [-3, 3]

        # Agregar componentes aleatorios en las direcciones
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(0, np.pi)

        x += np.sin(angle2) * np.cos(angle1) * move
        y += np.sin(angle2) * np.sin(angle1) * move
        z += np.cos(angle2) * move
        
        xf[i] = x
        yf[i] = y
        zf[i] = z

    return (x, y, z, xf, yf, zf)

# Parámetros para el generador congruencial lineal
seed = 42
a = 1664525
c = 1013904223
m = 2**32

step = 1000000

# Generar números pseudoaleatorios para los pasos
Rmove = linear_congruential_generator(seed, a, c, m, step)

# Simular el movimiento de la rana en 3 dimensiones
walk = random_walk_3d(0, 0, 0, step, Rmove)
print(f"\nDistancia del origen = {(walk[0]**2 + walk[1]**2 + walk[2]**2)**0.5} Unidades")

# Graficar la caminata de la rana en 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(walk[3], walk[4], walk[5], label='Caminata de la rana', color='blue')
ax.scatter(walk[3][-1], walk[4][-1], walk[5][-1], color='red', label='Posición final')
ax.set_xlabel('Posición en x')
ax.set_ylabel('Posición en y')
ax.set_zlabel('Posición en z')
ax.legend()
plt.title('Caminata de la rana en 3D')
plt.show()
