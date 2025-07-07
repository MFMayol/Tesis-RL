import pandas as pd

class Resultados:
    ''' Clase que permite leer un archivo de excel y devolverlo como un dataframe de pandas. '''
    def __init__(self, nombre_archivo):
        ''' Inicializa la clase con el nombre del archivo a leer. 
        Parámetros:
                nombre_archivo (str): Nombre del archivo a leer.
        '''
        self.nombre_archivo = nombre_archivo
    
    def leer_archivo(self):
        ''' Lee el archivo xlsx y devuelve un df del excel'''
        try:
            df = pd.read_excel(self.nombre_archivo)
            return df
        except FileNotFoundError:
            print(f"No se encontró el archivo '{self.nombre_archivo}'. Asegúrate de que esté en la misma carpeta que el script.")
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")