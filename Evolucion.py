import numpy as np # Para trabajar con matrices
import matplotlib.pyplot as plt # Para imprimir resultados
from mpl_toolkits import mplot3d # Imprimir 3Ds
from pymoo.model.problem import Problem # Definir el problema en pymoo
from pymoo.algorithms.nsga2 import NSGA2 # ALgoritmo para la evolución
from pymoo.optimize import minimize # Para realizar la evolución
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination

'''
Definición del cromosoma
Las compuertas not siempre son de 1 entrada, así que los valores de las not
mantiene constante y no se toma en cuenta para el cromosoma

cromosoma = (x1, x2, x3)
Alelo para x1 y x2: valores enteros restringidos a [1, 2, 3]
Alelo para x3: valores enteros restringidos a [0, 1]

Donde:
x1: Está relacionado a las compuertas AND
-> La cantidad de entradas de la AND se obtiene de la forma 2^(x1)

x2: Está relacionado a las compuertas OR
-> La cantidad de entradas de la OR se obtiene de la forma 2^(x2)

x3: Está relacionado al modelo (fenotipo)
-> 0: Lógica de min-términos
-> 1: Lógica de max-términos
'''

# Se define el problema por clase de forma vectorial
'''
Nota:
Funciones objetivas: todas deben estar planteadas para minimizarse

Integrados: A continuación se presentan carácteristicas relevantes sobre
los integrados utilizados, se consideró una alimentación de 5Vdd

-> Compuerta NOT
--> 6 compuertas por integrado
--> Coste: 1 unidad
--> Tiempo de respuesta: 10ns
--> Energía: 1 unidad
================================
-> Compuerta AND - 2 entradas
--> 4 compuertas por integrado
--> Coste: 1 unidad
--> Tiempo de respuesta: 12ns
--> Energía: 1 unidad

-> Compuerta AND - 4 entradas
--> 2 compuertas por integrado
--> Coste: 2.5 unidades
--> Tiempo de respuesta: 10ns
--> Energía: 2.5 unidades

-> Compuerta AND - 8 entradas
--> 1 compuerta por integrado
--> Conste: 6.25 unidades
--> Tiempo de respuesta: 150ns
--> Energía: 6.25 unidades
================================
-> Compuerta OR - 2 entradas
--> 4 compuertas por integrado
--> Coste: 1 unidad
--> Tiempo de respuesta:10ns
--> Energía: 1 unidad

-> Compuerta OR - 4 entradas
--> 2 compuertas por integrado
--> Coste: 2.5 unidades
--> Tiempo de respuesta: 125ns
--> Energía: 2.5 unidades

-> Compuerta OR - 8 entradas
--> 1 compuerta por integrado
--> Conste: 6.25 unidades
--> Tiempo de respuesta: 10ns
--> Energía: 6.25 unidades
'''

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, # Número de variables a optimizar
                         n_obj=3, # Número de funciones objetivas
                         n_constr=0,# Número de limitaciones
                         xl=np.array([1, 1, 0]), # Valor mínimo del alelos
                         xu=np.array([3, 3, 1]), # Valor máximo del alelo
                         type_var=int, # Trabajar con números enteros
                         elementwise_evaluation=True) # Evaluar elemento por elemento

    # Función para obtener la cantidad mínima de compuertas
    # inp: Cantidad de entradas totales
    # inte: Cantidad de entradas del integrado
    def _cantCircuitos(self, inp, inte):
        tot = 0 # Contiene la cantidad de compuertas a utilizar
        capas = 0 # Se utiliza para calcular el tiempo de propagación
        cond = True # Condición de final del ciclo
        temp = inp
        while cond:
            capas += 1
            if temp <= inte:
                # Se detiene el ciclo
                cond = False
                # A final se debe agregar un integrado, esí que se suma 1
                tot += 1
            else:
                # Cuantas salidas sobran
                mod = temp % inte
                # Cantidad de salidas al poner los integrados
                temp = temp // inte
                # Se suman la cantidad de integrados
                tot += temp
                # Se suman las salidas de los integrados más las salidas que faltan
                temp += mod
        if inp == 5: # Si son 5 entradas se trata de la capa intermedia
            tot *= 8
        return tot, capas

    # Función para la evaluación del algoritmo
    def _evaluate(self, x, out, *args, **kwargs):

        if x[2] == 0: # Se trabaja con min-términos
            AND = self._cantCircuitos(5, 2 ** x[0])
            OR = self._cantCircuitos(8, 2 ** x[1])
        elif x[2] == 1: # Se trabaja con max-términos
            AND = self._cantCircuitos(8, 2 ** x[0])
            OR = self._cantCircuitos(5, 2 ** x[1])
        else:
            print("Error")
            AND = [100, 5]
            OR = [100, 5]

        # Cantidad de compuertas por integrado, dependiendo de la cantidad de entradas
        inte = [4, 2, 1]
        # Coste y energía para la función de calidad
        cost = [1, 2.5, 6.25]

        # Tiempos de las compuertas en ns, tomadas de las bases de datos
        tiempoAND = [12, 10, 150]
        tiempoOR = [10, 125, 10]

        # Se obtiene la cantidad de compuertas AND
        cantAND = AND[0]
        capAND = AND[1]

        # Se obtiene la cantidad de compuertas OR
        cantOR = OR[0]
        capOR = OR[1]

        # Se obtiene la cantidad de integrados a utilizar
        cantInteAND = cantAND // inte[x[0] - 1]
        if cantAND % inte[x[0] - 1] > 0: # Se verifica si hay un residuo en la división
            cantInteAND += 1

        cantInteOR = cantOR // inte[x[1] - 1]
        if cantOR % inte[x[1] - 1] > 0: # Se verifica si hay un residuo en la división
            cantInteOR += 1

        # Función objetiva: coste
        f1 = cantInteAND * cost[x[0]-1] + cantInteOR * cost[x[1] - 1] + 2 * 1

        # Función objetiva: tiempo de establecimiento
        f2 = 2 * 10 + capAND * tiempoAND[x[0] - 1] + capOR * tiempoOR[x[1] - 1]

        # Función objetiva: energía emitida
        f3 = cantInteAND * cost[x[0]-1] + cantInteOR * cost[x[1] - 1] + 2 * 1

        out["F"] = np.column_stack([f1, f2, f3])

problem = MyProblem()

# Obtener Algoritmo, se utiliza NSGA-II
algorithm = NSGA2(pop_size=10,
                  sampling=get_sampling("int_random"),
                  crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                  mutation=get_mutation("int_pm", eta=3.0),
                  n_offsprings=5,
                  eliminate_duplicates=True)

# Se definen 50 iteraciones en la evolución
termination = get_termination("n_gen", 50)

# Se realiza la evolución
res = minimize(problem,
               algorithm,
               save_history=True,
               termination=termination,
               callback=None,
               verbose=True)

print("Mejor soluciones encontradas:")
print(res.X)
print("Valor de las funciones objetivas:")
print(res.F)

x = res.F[:, 0]
y = res.F[:, 1]
z = res.F[:, 2]

# Se imprimen los puntos del frente de pareto
plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.view_init(30, 45)
ax.scatter3D(x, y, z)
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')

plt.show()
