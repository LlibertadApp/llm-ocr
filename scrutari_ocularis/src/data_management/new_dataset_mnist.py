import argparse
import os
import numpy as np
import struct
import random
import torch
import random
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import base64
from PIL import Image
from io import BytesIO
import h5py

def generate_composite_image(digits, dataset):
    # Selecciona aleatoriamente imágenes del dataset para cada dígito
    selected_images = []
    for digit in digits:
        if digit == -1:  # -1 indica un espacio en blanco
            blank_image = torch.ones((1, 28, 28)) * -1  # Imagen en blanco con valor mínimo después de la normalización
            selected_images.append(blank_image)
        else:
            # Convertir dataset.targets a una lista
            targets_list = dataset.targets.tolist()

            # Obtener todos los índices de un dígito específico
            indices = [i for i, target in enumerate(targets_list) if target == digit]

            # Seleccionar aleatoriamente un índice de la lista de índices
            index = random.choice(indices)

            selected_images.append(dataset[index][0])
    
    # Combina las imágenes seleccionadas en una nueva imagen
    composite_image = torch.cat(selected_images, dim=2)  # Concatenar a lo largo del eje de ancho
    
    # Convertir la imagen compuesta en escala de grises a RGB
    composite_image_rgb = composite_image.repeat(3, 1, 1)  # Repetir el canal en escala de grises 3 veces

    return composite_image_rgb

def generate_composite_dataset(dataset_origin, target_size=120000):
    # Número de veces que cada combinación de dígitos debe ser generada
    num_repeats = target_size // 999
    
    count_test = 0;
    # Inicializa dataset como un diccionario con listas vacías
    dataset = {num: [None for _ in range(num_repeats)] for num in range(1000)}
    for num in range(1000):  # Incluimos el 0
        digits = list(str(num).zfill(3))
        for repeats in range(num_repeats):
            # Lógica para decidir si agregar un cero o no para números menores a 100
            if num < 100:
                if random.choice([True, False]):  # Decidir aleatoriamente si agregar ceros o no
                    # Generar imágenes con el dígito 0
                    if num < 10:
                        elemento_with_zero = generate_composite_image([-1, -1, int(digits[2])], dataset_origin)
                        dataset[num][repeats] = elemento_with_zero.permute(1, 2, 0).numpy()
                    else:
                        elemento_with_zero = generate_composite_image([-1, int(digits[1]), int(digits[2])], dataset_origin)
                        dataset[num][repeats] = elemento_with_zero.permute(1, 2, 0).numpy()
                else:
                    # Generar imágenes sin el dígito 0
                    elemento_without_zero = generate_composite_image([int(digits[0]), int(digits[1]), int(digits[2])], dataset_origin)
                    dataset[num][repeats] = elemento_without_zero.permute(1, 2, 0).numpy()
            else:
                elemento = generate_composite_image([int(digits[0]), int(digits[1]), int(digits[2])], dataset_origin)
                dataset[num][repeats] = elemento.permute(1, 2, 0).numpy()
        print("Ya genero la imagen:", num)
        """ 
        if count_test >= 100:
            print("Dimensiones de la imagen:", dataset[num][repeats].shape)
            print("Valor mínimo:", dataset[num][repeats].min())
            print("Valor máximo:", dataset[num][repeats].max())
            print("Tipo de datos:", dataset[num][repeats].dtype)
            print("Primeros píxeles:", dataset[num][repeats][0, :10])
            plt.imshow(dataset[num][repeats], cmap='gray')
            plt.title(f'Número: {num}, Repetición: {repeats}, digit: {digits}')
            plt.axis('off')
            plt.show()        
            count_test = 0
        else:
            count_test = count_test + 1 """

    return dataset
            
# Visualiza la imagen
""" 
    count_test = 0;
if count_test >= 100:
    plt.imshow(dataset[num][repeats], cmap='gray')
    plt.title(f'Número: {num}, Repetición: {repeats}, digit: {digits}')
    plt.axis('off')
    plt.show()
    count_test = 0
else:
    count_test = count_test + 1 """

def save_mnist(dataset, img_filename):
    # Aplanar la lista de imágenes y convertirlas a arrays de NumPy
    images = [np.array(img) for sublist in dataset.values() for img in sublist]

    # Cabeceras
    num_images = len(images)
    channels, rows, cols = images[0].shape

    # Escribir archivo de imágenes
    with open(img_filename, 'wb') as f:
        # Cabecera: magic number, número de imágenes, filas y columnas
        f.write(struct.pack('>IIII', 2051, num_images, rows, cols))
        
        # Imágenes
        for img in images:
            # Si las imágenes son en escala de grises y tienen una dimensión extra, la eliminamos
            img = img.squeeze()
            f.write(img.astype(np.uint8).tobytes())

def load_mnist(img_filename):
    """
    Carga imágenes desde un archivo en el formato MNIST.

    Args:
    - img_filename (str): Ruta al archivo de imágenes MNIST.

    Returns:
    - numpy.ndarray: Array de imágenes cargadas.
    """
    with open(img_filename, 'rb') as f:
        # Leer y desempaquetar la cabecera del archivo
        _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        # Leer el contenido del archivo y convertirlo en un array de NumPy
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols, 3)
    
    # Asegurarse de que las imágenes tienen la forma correcta
    assert images.shape == (num_images, rows, cols, 3), "Las imágenes cargadas no tienen la forma correcta."
    
    return images

def save_images_as_json(dataset, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for num, images_list in dataset.items():
        for repeats, image in enumerate(images_list):
            image_np = image
            # Convertir la imagen numpy en un objeto PIL
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            buffered = BytesIO()
            image_pil.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data = {
                'number': num,
                'repeats': repeats,
                'image_base64': image_base64
            }
            filename = os.path.join(folder_name, f"{num}_{repeats}.json")
            with open(filename, 'w') as f:
                json.dump(data, f)

def load_images_from_json(folder_name):
    dataset = {}

    # Listar todos los archivos JSON en la carpeta
    files = [f for f in os.listdir(folder_name) if f.endswith('.json')]

    for filename in files:
        filepath = os.path.join(folder_name, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

            # Decodificar la imagen desde base64
            img_data = base64.b64decode(data['image_base64'])
            image_pil = Image.open(BytesIO(img_data))
            image_np = np.array(image_pil) / 255.0

            num = data['number']
            repeats = data['repeats']

            if num not in dataset:
                dataset[num] = {}
            dataset[num][repeats] = torch.tensor(image_np).permute(2, 0, 1)

    return dataset

def save_images_as_hdf5(dataset, filename):
    with h5py.File(filename, 'w') as hf:
        for num, images_list in dataset.items():
            for repeats, image in enumerate(images_list):
                # Crear un nombre único para el grupo basado en el número y la repetición
                group_name = f"{num}_{repeats}"
                group = hf.create_group(group_name)
                
                # Guardar la imagen y la información en el grupo
                group.create_dataset("image", data=image)
                group.attrs["number"] = num
                group.attrs["repeats"] = repeats

def load_images_from_hdf5(filename):
    dataset = {}
    with h5py.File(filename, 'r') as hf:
        for group_name in hf.keys():
            group = hf[group_name]
            image_np = group["image"][()]
            
            # Extraer el número y la repetición de los atributos del grupo
            num = group.attrs["number"]
            repeats = group.attrs["repeats"]
            
            if num not in dataset:
                dataset[num] = {}
            dataset[num][repeats] = torch.tensor(image_np).permute(2, 0, 1)
    return dataset

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='New MNIST custom dataset')
    parser.add_argument('--view', type=int, default=0, metavar='N',
                        help='view n (default: 5) result random')
    parser.add_argument('--regenerate', default=True, action='store_true',
                        help='regenerate the dataset even if it exists (default: false)')
    args = parser.parse_args()

    # Si el dataset ya está generado y guardado y no se especifica --regenerate, simplemente lo cargamos
    if not args.regenerate and os.path.exists('custom_train-images-idx3-ubyte') and os.path.exists('custom_train-labels-idx1-ubyte'):
        train_composite_images = load_mnist('custom_train-images-idx3-ubyte', 'custom_train-labels-idx1-ubyte')
    else:
        # Transformación para MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Descargar y cargar MNIST
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)

        # Genera el dataset compuesto
        train_composite_images = generate_composite_dataset(train_dataset, 60000)
        test_composite_images = generate_composite_dataset(test_dataset, 15000)
        # Loguea cuántos números se generaron
        print(f"Se generaron {len(train_composite_images)} números en el dataset de entrenamiento.")
        print(f"Se generaron {len(test_composite_images)} números en el dataset de testeo.")
        
        # Guarda el dataset en el formato MNIST
        save_images_as_hdf5(train_composite_images, 'train_data_s.h5')
        save_images_as_hdf5(test_composite_images, 'test_data_s.h5')

if __name__ == '__main__':
    main()
