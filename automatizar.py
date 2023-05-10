import subprocess
from concurrent.futures import ThreadPoolExecutor

def ejecutar_script(nombre_script):
    print(f"Ejecutando {nombre_script}...")
    resultado = subprocess.run(["python", nombre_script])
    if resultado.returncode == 0:
        print(f"{nombre_script} ejecutado correctamente.")
    else:
        print(f"Error al ejecutar {nombre_script}. Código de error: {resultado.returncode}")

def main():
    scripts = ["QGA_BCQO_sim.py","modularizando_qga.py", "lineas.py"]
    #todo al mismo tiempo: 
    """
    with ThreadPoolExecutor() as executor:
        executor.map(ejecutar_script, scripts)"""
    for script in scripts:
        ejecutar_script(script)

if __name__ == "__main__":
    main()
