# Mock de BaseHandler para pruebas locales
class BaseHandlerMock:
    def __init__(self):
        # Configuraciones de inicialización básicas
        self.initialized = False

    def initialize(self, context):
        # Simula la inicialización del contexto
        self.initialized = True
        # Aquí puedes agregar la lógica de inicialización de tu modelo si es necesario