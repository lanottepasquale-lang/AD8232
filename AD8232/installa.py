import os

with open("requirements.txt", "r") as file:
    for pacchetto in file.read().splitlines():
        if pacchetto:  # Controlla che la riga non sia vuota
            print(f"Installando {pacchetto}...")
            os.system(f"mpremote mip install --target lib {pacchetto}")