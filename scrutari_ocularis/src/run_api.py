import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Obtener la configuración del entorno o establecer valores predeterminados
host = os.getenv('HOST', '0.0.0.0')
port = int(os.getenv('PORT', '8000'))
reload = os.getenv('RELOAD', 'false').lower() in ['true', '1', 't', 'yes']

if __name__ == "__main__":
    if reload:
        import sys
        # Asegurarse de que la raíz del proyecto está en sys.path
        ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(ROOT_PATH)
        # Ejecutar con la cadena de importación para habilitar la recarga
        uvicorn.run("api.main:app", host=host, port=port, reload=reload)
    else:
        # Ejecutar con la aplicación importada para un entorno de producción
        from api.main import app
        uvicorn.run(app, host=host, port=port)
