"""import numpy as np
import matplotlib.pyplot as plt

step = 40

Rmove = np.random.rand(step)
print(Rmove)
fmove = np.zeros(shape=(step))
print(fmove)

move = 0

for i in range (step):
    if Rmove[i] > 0.5:
        move += 1
        fmove[i] = move
    else:
        move -= 1
        fmove[i] = move
        print (move)
        
        plt.plot(fmove, "g")"""

import numpy as np
import matplotlib.pyplot as plt

def linear_congruential_generator(seed, a, c, m, n):
    random_numbers = np.zeros(n)
    random_numbers[0] = seed

    for i in range(1, n):
        random_numbers[i] = (a * random_numbers[i - 1] + c) % m

    return random_numbers / m

# Parámetros para el generador congruencial lineal
seed = 42
a = 1664525
c = 1013904223
m = 2**32

step = 1000000

Rmove = linear_congruential_generator(seed, a, c, m, step)
fmove = np.zeros(shape=(step))

move = 0

# Listas para almacenar datos para las gráficas
positions = []
random_frequencies = []

for i in range(step):
    if Rmove[i] > 0.5:
        move += 1
        fmove[i] = move
    else:
        move -= 1
        fmove[i] = move

    # Imprimir posición para cada paso en la consola
    print(f"Iteración {i + 1} posición {fmove[i]}")

    # Almacenar datos para las gráficas
    positions.append(fmove[i])
    random_frequencies.append(Rmove[i])

# Plot the positions of the rana
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(step), positions, "g")
plt.title('Posición de la rana en cada paso')
plt.xlabel('Paso')
plt.ylabel('Posición')

# Plot the random frequencies
plt.subplot(1, 2, 2)
plt.plot(range(step), random_frequencies, "b")
plt.title('Frecuencias de salida en cada paso')
plt.xlabel('Paso')
plt.ylabel('Frecuencia de salida')

plt.tight_layout()
plt.show()
