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



