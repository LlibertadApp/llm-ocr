#!/bin/bash

# Parar en el primer error
set -e

# Actualizar lista de paquetes e instalar prerequisitos
sudo apt-get update
sudo apt-get install -y python3 python3-pip git

# Clonar el repositorio (o puedes descomprimir un archivo con el código fuente)
git clone https://github.com/LlibertadApp/llm-ocr.git /var/www/scrutari_ocularis

# Ir al directorio de la aplicación
cd /var/www/scrutari_ocularis

# Instalar dependencias de Python
pip3 install -r requirements.txt

# Configurar variables de entorno
export ENV_VAR_NAME=value

# Asignar permisos (si es necesario)
chmod +x /var/www/scrutari_ocularis/src/run_api.py

# Iniciar la aplicación usando Uvicorn con Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker run_api:app
