import math
import matplotlib.pyplot as plt
from collections import defaultdict

def comparar_tiempos(tiempo1, tiempo2):
    if math.isclose(tiempo1, tiempo2):
        return "Tiempos iguales"
    elif tiempo1 > tiempo2:
        return "Modulos más lento"
    else:
        return "Original 2 más lento"

def leer_archivo(archivo):
    with open(archivo, "r") as f:
        lineas = f.readlines()
    return lineas

def procesar_lineas(lineas1, lineas2):
    tiempos1 = defaultdict(float)
    tiempos2 = defaultdict(float)
    
    for linea1, linea2 in zip(lineas1, lineas2):
        if linea1.strip() == "" or linea2.strip() == "":
            print("Se encontró una línea vacía. Fin del proceso.")
            break

        categoria = linea1.split(":")[0].strip()
        tiempo1 = float(linea1.split(":")[-1].strip())
        tiempo2 = float(linea2.split(":")[-1].strip())

        tiempos1[categoria] += tiempo1
        tiempos2[categoria] += tiempo2

    return tiempos1, tiempos2

def generar_grafico(tiempos1, tiempos2):
    categorias = list(tiempos1.keys())
    valores1 = list(tiempos1.values())
    valores2 = list(tiempos2.values())

    x = range(len(categorias))
    ancho = 0.35

    fig, ax = plt.subplots()
    barras1 = ax.bar(x, valores1, ancho, label="Modulos")
    barras2 = ax.bar([i + ancho for i in x], valores2, ancho, label="Original")

    ax.set_ylabel("Tiempo de ejecución")
    ax.set_title("Tiempos de ejecución por categoría")
    ax.set_xticks([i + ancho / 2 for i in x])
    ax.set_xticklabels(categorias, rotation=45, ha="right")
    ax.legend()

    plt.show()

def main():
    lineas1 = leer_archivo("datos.txt")
    lineas2 = leer_archivo("datu.txt")
    tiempos1, tiempos2 = procesar_lineas(lineas1, lineas2)
    
    generar_grafico(tiempos1, tiempos2)

if __name__ == "__main__":
    main()

