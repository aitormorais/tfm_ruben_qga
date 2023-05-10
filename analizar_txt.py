import re
import matplotlib.pyplot as plt

archivo = open("datos.txt", "r")  # Abre el archivo en modo lectura
contenido = archivo.read()  # Lee el contenido del archivo
archivo.close()  # Cierra el archivo

# Extrae los tiempos de ejecución utilizando expresiones regulares
tiempos = re.findall(r"Tiempo de ejecucion de ([^:]+): ([\d.]+)", contenido)

# Crea un diccionario para almacenar los tiempos de ejecución por categoría
tiempos_dict = {}
for tiempo in tiempos:
    categoria = tiempo[0]
    valor = float(tiempo[1])
    if categoria in tiempos_dict:
        tiempos_dict[categoria].append(valor)
    else:
        tiempos_dict[categoria] = [valor]

# Encuentra la categoría con el tiempo de ejecución máximo
max_categoria = max(tiempos_dict, key=lambda k: sum(tiempos_dict[k]))

# Crea un gráfico de barras con los tiempos de ejecución por categoría
categorias = list(tiempos_dict.keys())
valores = [sum(tiempos_dict[categoria]) for categoria in categorias]

plt.bar(categorias, valores)
plt.xlabel("Categoría")
plt.ylabel("Tiempo de ejecución")
plt.title("Tiempos de ejecución por categoría")
plt.xticks(rotation=90)
plt.show()

print("La categoría con el tiempo de ejecución máximo es:", max_categoria)
