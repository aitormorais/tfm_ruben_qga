# Nombre de los archivos de entrada y salida
input_file = "modularizando_qga.py"
output_file = "archivo_salida.txt"

# Leer el contenido del archivo de entrada
with open(input_file, "r") as infile:
    contenido = infile.read()

# Eliminar espacios y tabulaciones
contenido_sin_espacios = contenido.replace(" ", "").replace("\t", "")

# Escribir el contenido en el archivo de salida
with open(output_file, "w") as outfile:
    outfile.write(contenido_sin_espacios)
