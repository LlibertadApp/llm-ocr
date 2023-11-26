import h5py
import numpy as np

class ElectionImageStorage:

    def __init__(self, filename):
        self.filename = filename

    def save(self, file_info, resultados_info):
        with h5py.File(self.filename, 'a') as hf:
            file_name = file_info['file_name']
            if file_name in hf:
                print(f"El archivo {file_name} ya ha sido procesado y guardado.")
                return False
            group = hf.create_group(file_name)
            for codigo, info in resultados_info.items():
                group.create_dataset(f"{codigo}_image", data=info['imagen'])
                group.attrs[f"{codigo}_votos"] = info['votos']
            return True

    def load(self, file_name):
        with h5py.File(self.filename, 'r') as hf:
            if file_name in hf:
                group = hf[file_name]
                data = {}
                for item in group.keys():
                    if item.endswith('_image'):  # Check if the item name ends with '_image'
                        codigo = item.rsplit('_image', 1)[0]  # Extract the codigo part of the name
                        data[codigo] = {
                            'imagen': np.array(group[item]),  # Access the dataset directly
                            'votos': group.attrs[f"{codigo}_votos"]  # Access the attribute directly, without the '_image' suffix
                        }
                return data
            else:
                print(f"El archivo {file_name} no se encuentra en el almacenamiento.")
                return None

    def load_all(self):
        all_data = {}
        with h5py.File(self.filename, 'r') as hf:
            for file_name in hf.keys():
                group = hf[file_name]
                all_data[file_name] = {}
                for item in group.keys():
                    # The following assumes that your dataset names within the group follow the pattern 'codigo_image'
                    if item.endswith('_image'):  # Check if the item name ends with '_image'
                        codigo = item.rsplit('_image', 1)[0]  # Extract the codigo part of the name
                        all_data[file_name][codigo] = {
                            'imagen': np.array(group[item]),  # Access the dataset directly
                            'votos': group.attrs[f"{codigo}_votos"]  # Access the attribute directly
                        }
        return all_data