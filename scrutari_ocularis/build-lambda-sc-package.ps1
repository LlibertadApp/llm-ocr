# Define los parámetros de configuración
$projectName = "segmentation_cropping"
$version = "1.0.0" # Asegúrate de cambiar esto según corresponda
$srcFolder = ".\segmentation_cropping" # Asegúrate de que la ruta es correcta
$buildFolder = ".\build"
$venvFolder = ".\venv"
$zipFile = "$projectName-$version.zip"

# Crear un nuevo entorno virtual
python -m venv $venvFolder

# Activar el entorno virtual
. "$venvFolder\Scripts\Activate.ps1"

# Instalar dependencias en el entorno virtual
pip install -r "$srcFolder\requirements.txt"

# Desactivar el entorno virtual para copiar los archivos
deactivate

# Crear un nuevo directorio de construcción
New-Item -Path $buildFolder -ItemType Directory -Force

# Copiar el código fuente al directorio de construcción
Copy-Item "$srcFolder\*" -Destination $buildFolder -Recurse -Force

# Copiar las dependencias del entorno virtual al directorio de construcción
Copy-Item "$venvFolder\Lib\site-packages\*" -Destination $buildFolder -Recurse -Force

# Crear archivo ZIP con todo el contenido del directorio de construcción
Compress-Archive -Path "$buildFolder\*" -DestinationPath $zipFile -Force
