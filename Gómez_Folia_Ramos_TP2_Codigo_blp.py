##############################################################################
#                          Universidad de San Andres                         #
#                           Métodos Econométricos                            #  
#                             Trabajo Practico 2                             #
#              Autores: Facundo Gómez, Tomás Folia y Julian Ramos            #
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ttest_ind
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.kernel_regression import KernelReg as kr
from matplotlib.pyplot import figure
from scipy.stats import norm
import multiprocessing
from joblib import Parallel, delayed
from scipy.optimize import minimize
import itertools
import random
from tqdm import tqdm
np.random.seed(444) 
random.seed(444) 

#%%
# Leer el archivo .dta
df = pd.read_stata("C:/Users/Usuario/Desktop/Trabajo - UdeSA/Maestría 2024/Cursada/Segundo trimestre/MÉTODOS ECONOMÉTRICOS Y ORGANIZACIÓN INDUSTRIAL APLICADA/TPs-metodos/TP2/input/data_blp_final.dta")

#Creamos una función para generar los precios aleatorios que luego utilizaremos:
def generar_precios_aleatorios(fila, df, num_instruments=30):
    # Filtrar el DataFrame para obtener las demás tiendas
    otras_tiendas = df[(df['semana'] == fila['semana']) &
                       (df['marca'] == fila['marca']) &
                       (df['tienda'] != fila['tienda'])]
    
    #Obtenemos una lista única de tiendas:
    tiendas_unicas = otras_tiendas['tienda'].unique()
    
    #Si hay suficientes tiendas únicas (que siempre las hay), 
    #este código selecciona tiendas aleatorias sin repetición:
    if len(tiendas_unicas) >= num_instruments:
        tiendas_seleccionadas = random.sample(list(tiendas_unicas), num_instruments)
        # Obtener los precios de las tiendas seleccionadas
        precios_aleatorios = []
        for tienda in tiendas_seleccionadas:
            precio_tienda = otras_tiendas[otras_tiendas['tienda'] == tienda]['precio'].values
            # Asegurarse de que la tienda seleccionada tenga un precio
            if len(precio_tienda) > 0:
                precios_aleatorios.append(precio_tienda[0])
        return precios_aleatorios
    else:
        return [np.nan] * num_instruments

#Inicializamos las columnas de precios aleatorios:
for i in range(1, 31):
    df[f'pricestore{i}'] = np.nan

#Esto es un código que, además de generar los datos de precios aleatorios,
#incorpora una barra de progreso para ver el avance de la ejecución del código:
for idx, fila in tqdm(df.iterrows(), total=df.shape[0], desc="Generando precios aleatorios"):
    precios_aleatorios = generar_precios_aleatorios(fila, df)
    for i in range(1, 31):
        df.at[idx, f'pricestore{i}'] = precios_aleatorios[i-1] if i-1 < len(precios_aleatorios) else np.nan
        
#Guardamos el DataFrame resultante como un archivo .dta para utilizarlo en el loop interno:
df.to_stata("C:/Users/Usuario/Desktop/Trabajo - UdeSA/Maestría 2024/Cursada/Segundo trimestre/MÉTODOS ECONOMÉTRICOS Y ORGANIZACIÓN INDUSTRIAL APLICADA/TPs-metodos/TP2/input/data_blp_final2.dta", write_index=False)


#%% CREAMOS OTRAS VARIABLES IMPORTANTES
data = pd.read_stata("C:/Users/Usuario/Desktop/Trabajo - UdeSA/Maestría 2024/Cursada/Segundo trimestre/MÉTODOS ECONOMÉTRICOS Y ORGANIZACIÓN INDUSTRIAL APLICADA/TPs-metodos/TP2/input/data_blp_final2.dta")

#Generamos la variable "brand_id" de acuerdo a la tabla provista en la consigna
#sobre las marcas del mercado:
def assign_branded_id(marca):
    if marca in [1, 2, 3]:
        return 1
    elif marca in [4, 5, 6]:
        return 2
    elif marca in [7, 8, 9]:
        return 3
    elif marca in [10, 11]:
        return 4
    else:
        return None  #En caso de que el valor de marca no esté en el rango esperado

#Luego, definimos la función que asigna el valor de "is_branded" para generar esta
#variable:
def assign_is_branded(marca):
    if marca in [10, 11]:
        return 0
    else:
        return 1
    
    
#Corregimos los market_share a nivel de las 4 grandes marca, denotando esta variable
#como "big_shares". También general el logaritmo de estos valores y los nuevos delta observados,
#ya que estas variables serán utilizadas para el inciso (4.b):    
def calculate_shares(df):
    # Agrupamos por 'semana', 'tienda' y 'marca4' y calculamos las nuevas variables:
    df['big_shares'] = df.groupby(['semana', 'tienda', 'brand_id'])['real_ms_per_brand_and_store'].transform('sum')
    df['log_big_shares'] = np.log(df['big_shares'])
    df['delta_jpt_big'] = df['log_big_shares'] - np.log(0.36)
    return df

#Por último, aplicamos las funciones al DataFrame para crear las nuevas columnas:
data['brand_id'] = data['marca'].apply(assign_branded_id)
data['is_branded'] = data['marca'].apply(assign_is_branded)
data['store_week_id'] = pd.factorize(data[['tienda', 'semana']].apply(tuple, axis=1))[0]
#Aplicamos la función para los "big shares":
data = calculate_shares(data)
print(data.head())

#Cambiamos el nombre de la variable que indica el market share por marca y tienda:
data.rename(columns={'real_ms_per_brand_and_store': 'market_share'}, inplace=True)


#%% LOOP INTERNO y EXTERNO:
cores = 6

# Generar distribuciones de ingresos
incomeDists = []
for i in range(1, 20):
    temp = 'income_dist' + str(i)
    # Convertimos cada columna de ingresos a un array y lo agregamos a la lista
    array = np.array(data[temp])
    incomeDists.append(array)
# Concatenamos todas las distribuciones de ingresos en un solo array y obtenemos los valores únicos
incomeDistTotal = np.unique(np.hstack(incomeDists))

# Seleccionamos un subconjunto de distribuciones de ingresos para la simulación
incomeDistTotal2 = np.random.choice(incomeDistTotal, size=10)

# Paso 1: Calcular los market shares
# shares+sharesPar = Fast Integral Solver
# La función shares utiliza a sharesPar para realizar cálculos en paralelo de los valores de las participaciones de mercado.
# Armamos matrices con sI al multiplicar por los precios y de sB al mantenerlo según si es un producto de marca.
def shares(incomes, vVec, d, sB, sI, df):
    d = np.array(d)  # Convertimos a matriz NumPy
    df['sum'] = 0
    init1 = np.transpose(np.tile(sB*df['is_branded'], (len(vVec)*len(incomes), 1))) 
    init2 = np.transpose(np.tile(sI*df['precio'], (len(vVec)*len(incomes), 1)))
    vVecLong = np.concatenate([vVec for i in incomes])
    incomesLong = np.repeat(incomes, len(vVec))
    top = d[:, np.newaxis] * np.exp(init1 * vVecLong) * np.exp(init2 * incomesLong)
    temp = pd.concat([df[['store_week_id']], pd.DataFrame(top, index=df.index, dtype=float)], axis=1)
    sums = temp.groupby('store_week_id').sum() + 1
    sumsArray = sums.iloc[np.repeat(np.arange(len(sums)), 11)]
    df['sum'] = df['sum'] + (1 / (len(vVec) * len(incomes))) * np.array(top / sumsArray).sum(axis=1)
    return df['sum']

def sharesPar(d, sB, sI, df):
    iVec = incomeDistTotal2
    vVec = np.random.normal(size=10)
    split1 = np.hsplit(np.array(iVec), 5)
    split2 = np.hsplit(vVec, 10)
    interim = []
    for s1 in split1:
        for s2 in split2:
            interim.append([s1, s2])
    if __name__ == "__main__":
        totals = Parallel(n_jobs=cores)(delayed(shares)(i[0], i[1], d, sB, sI, df) for i in interim)
    new = 0
    for p in totals:
        new = new + p
    return new / len(interim)

# Función deltaNext2 para obtener delta
# Esta función calcula un vector de ajuste basado en la diferencia entre las participaciones calculadas en la función
# sharesPar y las participaciones predichas. En conjunto, las tres funciones (las de shares y delta) forman un proceso
# iterativo en el que se utilizan los cálculos de participación en el mercado para ajustar los valores de utilidad esperada.
def deltaNext2(d, sB, sI, df):
    predicted = sharesPar(d, sB, sI, df)
    interim = d * (df.big_shares / predicted)
    return interim

# Paso 2: Regresión de variables instrumentales
# Crear dummies de marca para la matriz de características
brandDummies = pd.get_dummies(data['brand_id'], prefix='brand_id')
brandDummies = brandDummies.astype('float64')

# Creamos la matriz de características incluyendo precio, promocion y dummies de marca
Xmat = np.array(data[['precio', 'descuento']].join(brandDummies))
XmatT = Xmat.transpose()

# Crear matriz de instrumentos que incluye  el precio mayorista (costo), el precio
#promedio en otros mercados y los precios en otros 30 mercados:
instrumentList = ['costo', 'hausman_iv']
for i in range(1, 31):
    string = 'pricestore' + str(i)
    instrumentList.append(string)

Zmat = np.array(data[instrumentList]).astype('float64')
ZmatT = Zmat.transpose()

# Calcular la matriz Omega para GMM
# Utilizamos variables instrumentales para abordar la endogeneidad en la variable de precio
omega = np.linalg.inv(np.dot(ZmatT, Zmat))
# Calculamos la primera parte de la matriz para la estimación 2SLS
first = np.linalg.inv(np.dot(np.dot(np.dot(np.dot(XmatT, Zmat), omega), ZmatT), Xmat))
# Calculamos la segunda parte de la matriz para la estimación 2SLS
second = np.dot(np.dot(np.dot(XmatT, Zmat), omega), ZmatT)
# Multiplicamos ambas partes para obtener la matriz de pre-multiplicación
preMultMat = np.dot(first, second)

# Estimar parámetros lineales usando 2SLS (mínimos cuadrados en dos etapas):
def ivEval(d, df):
    dMat = np.array(d)
    # Estimamos los coeficientes beta
    beta = np.dot(preMultMat, dMat)
    # Calculamos el término de error xi
    xi = dMat - np.dot(Xmat, beta)
    # Calculamos la parte externa para el criterio de optimización:
    outer = np.dot(ZmatT, xi)
    # Calculamos el criterio de optimización como el valor absoluto del producto de las matrices
    return np.abs(np.dot(np.dot(outer.transpose(), omega), outer))

# Paso 3:
# Iterar para resolver delta 
# Utilizamos el método de mapeo de contracción para ajustar los market share simulados a los observados. 
# Se ajusta delta hasta que los market share simulados sean lo mas parecidos a los observados.
def BLP(sigma_array):
    sigmaB = sigma_array[0]
    sigmaI = sigma_array[1]
    data['delta'] = 1 / 12
    delta = 0
    deltaNew = np.exp(data['delta'])
    counter = 0
    with tqdm(total=50, desc="Iteraciones de BLP") as pbar:
        while np.linalg.norm(deltaNew - delta) > 1 and counter < 50: #Límite de iteraciones
            counter += 1
            delta = deltaNew
            deltaNew = deltaNext2(delta, sigmaB, sigmaI, data)
            pbar.update(1)
    delta = np.log(deltaNew)
    criterion = ivEval(delta, data)
    data['calculated_delta'] = delta
    return criterion

# Función para estimar beta y alpha
def estimate_beta_alpha(delta, df):
    beta_alpha = np.dot(preMultMat, delta)
    return beta_alpha

# Función de callback para imprimir los valores de cada iteración
def print_iteration(xk):
    print(f"Current parameters: {xk}")
    
# Aplicamos la función minimize para encontrar los valores de los parámetros que minimizan el criterio GMM.
init = np.array([0.1, 0.1]) 
test = minimize(BLP, init, method='L-BFGS-B', callback=print_iteration)
result = test.x
print(test)

#%%
# Obtener el vector delta final
sigmaB, sigmaI = result
data['delta'] = 1 / 12
delta = 0
deltaNew = np.exp(data['delta'])
counter = 0
with tqdm(total=50, desc="Iteraciones finales de delta") as pbar:
    while np.linalg.norm(deltaNew - delta) > 1 and counter < 50:
        counter += 1
        delta = deltaNew
        deltaNew = deltaNext2(delta, sigmaB, sigmaI, data)
        pbar.update(1)
delta = np.log(deltaNew)

# Estimar beta y alpha
beta_alpha = estimate_beta_alpha(delta, data)
print("Beta and Alpha estimates:", beta_alpha)

coef_names = [r'$\sigma_b$', r'$\sigma_I$', 'price', 'promotion'] + list(brandDummies.columns)
estimates = np.append([sigmaB, sigmaI], beta_alpha)
table = pd.DataFrame({'Coefficient': coef_names, 'Estimate': estimates})
print(table)

data.to_stata("C:/Users/Usuario/Desktop/Trabajo - UdeSA/Maestría 2024/Cursada/Segundo trimestre/MÉTODOS ECONOMÉTRICOS Y ORGANIZACIÓN INDUSTRIAL APLICADA/TPs-metodos/TP2/input/data_blp_loopf.dta", write_index=False)




