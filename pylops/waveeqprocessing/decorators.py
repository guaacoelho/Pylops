from functools import wraps

def update_op_coords_if_needed(method):
    """
    Se o operador possuir segyReader inicializado significa que está executando
    a partir da utilização de dados de um arquivo SEGY.

    Sendo assim, é necessário que o operador seja atualizado com os parâmetros corretos
    para o shot que ele está executando
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "segyReader"):
            self._update_op_coords()
        return method(self, *args, **kwargs)
    return wrapper