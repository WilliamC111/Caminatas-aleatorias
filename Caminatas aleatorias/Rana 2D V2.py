import numpy as np
import matplotlib.pyplot as plt

def linear_congruential_generator(seed, a, c, m, n):
    random_numbers = np.zeros(n)
    random_numbers[0] = seed

    for i in range(1, n):
        random_numbers[i] = (a * random_numbers[i - 1] + c) % m

    return random_numbers / m

def random_walk(xi, yi, n, Rmove, target_position):
    x, y = xi, yi
    xf, yf = np.zeros(n), np.zeros(n)

    for i in range(n):
        ri = Rmove[i]
        move = int(ri * 4)
        
        # Agregar componente aleatoria en la dirección
        angle = np.random.uniform(0, 2 * np.pi)
        x += np.cos(angle) * move
        y += np.sin(angle) * move

        xf[i] = x
        yf[i] = y

        # Agregar condición para imprimir el número de iteraciones cuando la rana alcance la posición específica
        if (x, y) == target_position:
            print(f"La rana alcanzó la posición ({x}, {y}) en la iteración {i + 1}")
            break

    return (x, y, xf, yf)

# Parámetros para el generador congruencial lineal
seed = 42
a = 1664525
c = 1013904223
m = 2**32

step = 1000000

# Posición objetivo
target_position = (250, 300)

# Generar números pseudoaleatorios para los pasos
Rmove = linear_congruential_generator(seed, a, c, m, step)

# Simular el movimiento de la rana en 2 dimensiones
walk = random_walk(0, 0, step, Rmove, target_position)
print(f"\nDistancia del origen = {(walk[0]**2 + walk[1]**2)**0.5} Unidades")

# Graficar la caminata de la rana en 2D
plt.figure(figsize=(10, 5))
plt.plot(walk[2], walk[3], label='Caminata de la rana', color='blue')
plt.scatter(walk[2][-1], walk[3][-1], color='red', label='Posición final')
plt.title('Caminata de la rana en 2D')
plt.xlabel('Posición en x')
plt.ylabel('Posición en y')
plt.legend()
plt.show()

# Graficar la frecuencia de salida
plt.plot(range(step), Rmove, "b")
plt.xlabel('Paso')
plt.ylabel('Frecuencia de salida')
plt.title('Frecuencia de salida en cada paso')
plt.show()

