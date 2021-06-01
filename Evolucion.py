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

cromosoma = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
Alelo para x1, x2, ..., x9: valores enteros restringidos a [1, 2, 3]
Alelo para x10: valores enteros restringidos a [0, 1]

Donde:
x1, x2, ..., x9: Está relacionado con las compuertas

x10: Está relacionado al modelo (fenotipo)
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
        super().__init__(n_var=10, # Número de variables a optimizar
                         n_obj=3, # Número de funciones objetivas
                         n_constr=0,# Número de limitaciones
                         xl=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), # Valor mínimo del alelos
                         xu=np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 1]), # Valor máximo del alelo
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
        return tot, capas

    # Función para la evaluación del algoritmo
    def _evaluate(self, x, out, *args, **kwargs):
        totAND = []
        totOR = []
        capAND = []
        capOR = []
        if x[9] == 0: # Se trabaja con min-términos
            for i in range(len(x)-1):
                if i == 8:
                    temp = self._cantCircuitos(8, 2 ** x[i])
                    totOR.append(temp[0])
                    capOR.append(temp[1])
                else:
                    temp = self._cantCircuitos(5, 2 ** x[i])
                    totAND.append(temp[0])
                    capAND.append(temp[1])

        elif x[9] == 1: # Se trabaja con max-términos
            for i in range(len(x)-1):
                if i == 8:
                    temp = self._cantCircuitos(8, 2 ** x[i])
                    totAND.append(temp[0])
                    capAND.append(temp[1])
                else:
                    temp = self._cantCircuitos(5, 2 ** x[i])
                    totOR.append(temp[0])
                    capOR.append(temp[1])
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

        totAND = np.array(totAND)
        capAND = np.array(capAND)

        totOR = np.array(totOR)
        capOR = np.array(capOR)

        # Se obtiene la cantidad de integrados a utilizar
        totInteOR = [0, 0, 0]
        totInteAND = [0, 0, 0]
        if len(totAND) > 1:
            for i in range(len(totAND)):
                totInteAND[x[i]-1] += totAND[i]
            totInteOR[x[9]-1] += totOR[0]
        else:
            for i in range(len(totOR)):
                totInteOR[x[i]-1] += totOR[i]
            totInteAND[x[9]-1] += totAND[0]

        totORinte = []
        totANDinte = []
        for i in range(len(totInteOR)):
            temp = totInteOR[i] // inte[i]
            if totInteOR[i] % inte[i] > 0:
                temp += 1
            totORinte.append(temp)
            temp = totInteAND[i] // inte[i]
            if totInteAND[i] % inte[i] > 0:
                temp += 1
            totANDinte.append(temp)

        # Función objetiva: coste
        f1 = 2
        for i in range(len(cost)):
            f1 += cost[i] * (totANDinte[i] + totORinte[i])

        # Función objetiva: tiempo de establecimiento
        whereAND = np.where(capAND == np.amax(capAND))
        maxAND = np.amax(capAND)
        whereOR = np.where(capOR == np.amax(capOR))
        maxOR = np.amax(capOR)
        if len(capAND) > 1:
            f2 = 2 * 10 + maxAND * tiempoAND[x[whereAND[0][0]] - 1] + maxOR * tiempoOR[x[8] - 1]
        else:
            f2 = 2 * 10 + maxAND * tiempoAND[x[8] - 1] + maxOR * tiempoOR[x[whereOR[0][0]] - 1]

        # Función objetiva: energía emitida
        f3 = f1

        out["F"] = np.column_stack([f1, f2, f3])

problem = MyProblem()

# Obtener Algoritmo, se utiliza NSGA-II
algorithm = NSGA2(pop_size=3000,
                  sampling=get_sampling("int_random"),
                  crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                  mutation=get_mutation("int_pm", eta=3.0),
                  n_offsprings=5,
                  eliminate_duplicates=True)

# Se definen 50 iteraciones en la evolución
termination = get_termination("n_gen", 1000)

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
