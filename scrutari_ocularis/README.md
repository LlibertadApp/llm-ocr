# Scrutari Ocularis

El proyecto Scrutari Ocularis está diseñado para desarrollar un sistema OCR (Reconocimiento Óptico de Caracteres) altamente especializado, destinado a la interpretación y análisis automatizado de telegramas y actas de escrutinio electoral.

## El desarrollo de este sistema se estructura en tres fases fundamentales

* Desarrollo de un Modelo de Lenguaje de Aprendizaje Profundo (LLM): Este paso implica la definición y entrenamiento de un modelo de lenguaje avanzado que pueda entender y procesar el contenido textual específico de los documentos electorales.

* Construcción de un Dataset con un 99% de Confiabilidad: En esta etapa se genera un conjunto de datos altamente fiable, el cual servirá para entrenar y validar el modelo, asegurando su precisión en la detección y reconocimiento de caracteres y estructuras textuales relevantes.

* Creación de un Dataset Basado en las Actas de las Elecciones Generales: El propósito de este proceso es recopilar y organizar un dataset derivado directamente de las actas de elecciones generales, lo que permitirá al modelo adaptarse y especializarse en el contexto y la terminología específica de estos documentos.

## El objetivo final

Es desarrollar un sistema capaz de predecir y transcribir con precisión el contenido de nuevos telegramas provenientes de futuros procesos electorales, facilitando así la tarea de escrutinio y garantizando mayor eficiencia y transparencia en la contienda electoral.

## significado de ScrutariOcularis

**Scrutari** es una forma de **scrutiny** '*escrutinio*' en latín, y **Ocularis** implica *visión*, adecuado para OCR.

## Segmentación de Telegramas para la Generación de Dataset Electoral

Este módulo del proyecto Scrutari Ocularis se centra en el procesamiento y análisis detallado de imágenes de tablas estructuradas, con el objetivo principal de segmentar y extraer celdas individuales de las actas de elecciones generales. Mediante técnicas avanzadas de visión por computadora, se identifican y aíslan los componentes tabulares específicos dentro de los documentos digitalizados, lo que facilita la recopilación de datos precisos y su posterior incorporación en un dataset.

Cada celda segmentada es cuidadosamente recortada y preparada para su análisis OCR, garantizando la extracción fiable de la información textual que contienen. Este proceso meticuloso no solo mejora la calidad del dataset resultante sino que también optimiza la fase de entrenamiento del modelo LLM, proporcionando una base sólida para la predicción y reconocimiento efectivo de los datos en nuevos documentos de escrutinio.

La estrategia de segmentación es vital para superar los desafíos presentados por la diversidad de formatos y la calidad de las imágenes de las actas, asegurando que el sistema pueda operar con un alto grado de exactitud en condiciones reales. Con este enfoque, Scrutari Ocularis se prepara para convertirse en una herramienta esencial en el proceso de digitalización y análisis electoral, contribuyendo significativamente a la integridad y eficiencia del escrutinio.

[Ver el resultado en segmentation.ipynb](./segmentation.ipynb)

# segmentation_cropping

## Descripción

Este proyecto es una función de AWS Lambda para el alineamiento y recorte de imágenes, específicamente pensado para telegramas. La función ajusta las imágenes basándose en los parámetros recibidos, realiza un recorte inteligente y luego guarda los resultados en un bucket de Amazon S3.


## Características

- Alineamiento automático de imágenes basado en plantillas.
- Recorte de imágenes basado en áreas de interés.
- Almacenamiento de imágenes resultantes en Amazon S3.

## Configuración de Lambda

Antes de desplegar la función, asegúrate de configurar las siguientes variables de entorno en la consola de AWS Lambda o en tu archivo de configuración:

- `BUCKET_OCR_IMAGES`: Nombre del bucket de S3 para imágenes procesadas.
- `BOI_ENDPOINT_URL`: URL del endpoint para el servicio compatible con S3 (como MinIO).
- `BOI_ACCESS`: Clave de acceso para el servicio compatible con S3.
- `BOI_SECRET`: Clave secreta para el servicio compatible con S3.

Estas variables se utilizan para configurar el cliente de S3 dentro de la función de Lambda y deben ser establecidas para que la función pueda interactuar correctamente con el servicio de almacenamiento.

## Evento de Prueba

Un evento de prueba para la función Lambda puede ser configurado como sigue:

```json
{
  "code": "mock_telegrama_0",
  "image_path": "telegramas/mock_telegrama_0.jpg",
  "template_path": "templates/AE-150-23-telegrama.tiff"
}
```

Este evento simula una invocación de Lambda donde se proporciona la información necesaria para procesar la imagen del telegrama.

## Contexto de Prueba

Un contexto de prueba para simular la ejecución en un entorno local puede definirse en un script de Python como se muestra en el ejemplo de prueba proporcionado. Esto es útil para pruebas locales antes de desplegar la función en AWS.

## Entorno de Prueba

Para probar la función de Lambda localmente, puedes utilizar el Jupyter Notebook segmentation_cropping.ipynb. Este entorno de pruebas te permite cargar imágenes, ejecutar la función y visualizar los resultados sin necesidad de desplegar la función en AWS.

## Scripts de Construcción

El script `build-lambda-sc-package.ps1` se utiliza para preparar el paquete de despliegue de la función Lambda. Este script instala todas las dependencias, copia los archivos necesarios al directorio de construcción y crea un archivo `.zip` listo para ser desplegado en AWS Lambda.

Para ejecutar el script de construcción, utiliza el siguiente comando en PowerShell:

```powershell
./build-lambda-sc-package.ps1
```

