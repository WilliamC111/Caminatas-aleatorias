"""import random 
import numpy as np
import matplotlib.pyplot as plt

def random_walk(xi, yi, n):
    x, y = xi, yi
    xf, yf = np.zeros(shape=(n)), np.zeros(shape=(n))
    for i in range(n):
        (dx, dy) = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        x += dx
        xf[i] = x
        y += dy
        yf[i] = y
        print("Paso " + str(i + 1) + " = (x,y) = (" + str(x) + "," + str(y))
        plt.plot(xf, yf, "b")
    return (x, y)

walk = random_walk(0, 0, 50)
print("\nDistancia del origen = " + str((walk[0]**2 + walk[1]**2)**0.5) + " Unidades")"""
import numpy as np
import matplotlib.pyplot as plt

def linear_congruential_generator(seed, a, c, m, n):
    random_numbers = np.zeros(n)
    random_numbers[0] = seed

    for i in range(1, n):
        random_numbers[i] = (a * random_numbers[i - 1] + c) % m

    return random_numbers / m

def random_walk(xi, yi, n, Rmove):
    x, y = xi, yi
    xf, yf = np.zeros(n), np.zeros(n)

    for i in range(n):
        ri = Rmove[i]
        move = int(ri * 4)
        if move == 0:
            x += 1
        elif move == 1:
            x -= 1
        elif move == 2:
            y += 1
        elif move == 3:
            y -= 1
        
        xf[i] = x
        yf[i] = y

    return (x, y, xf, yf)

# Parámetros para el generador congruencial lineal
seed = 42
a = 1664525
c = 1013904223
m = 2**32

step = 1000000

# Generar números pseudoaleatorios para los pasos
Rmove = linear_congruential_generator(seed, a, c, m, step)

# Simular el movimiento de la rana en 2 dimensiones
walk = random_walk(0, 0, step, Rmove)
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
