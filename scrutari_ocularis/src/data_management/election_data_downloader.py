import os
import json
import requests

class ElectionDataDownloader:
    def __init__(self, folder_path, file_telegram):
        self.folder_path = folder_path
        self.file_telegram = file_telegram
        self.json_data = None

    def get_or_download_data(self):
        # Comprobar si la carpeta existe y si no, crearla
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # Componer el nombre del archivo JSON
        json_filename = f"{self.file_telegram}_scope_data.json"
        json_file_path = os.path.join(self.folder_path, json_filename)

        # Verificar si el archivo ya existe
        if os.path.isfile(json_file_path):
            # Leer el contenido del archivo JSON existente
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                self.json_data = json.load(json_file)
            print(f"Datos leídos desde {json_file_path}")
        else:
            # URL del endpoint de la API
            api_url = f"https://resultados.gob.ar/backend-difu/scope/data/getScopeData/{self.file_telegram}/1"
            # Realizar la llamada GET a la API
            response = requests.get(api_url)
            # Verificar que la llamada fue exitosa
            if response.status_code == 200:
                # Convertir la respuesta a JSON
                self.json_data = response.json()
                # Guardar la respuesta en un archivo JSON
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(self.json_data, json_file, ensure_ascii=False, indent=4)
                print(f"Datos guardados en {json_file_path}")
            else:
                print(f"Error al realizar la solicitud: {response.status_code}")

        return self.json_data