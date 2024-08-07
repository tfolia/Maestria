/*******************************************************************************
                          Universidad de San Andrés	         
                             Métodos Econométricos
							  Trabajo Práctico 2
							   Código Ej. 4 y 5
								  Julio 2024
					Facundo Gómez, Tomás Folia y Julian Ramos
********************************************************************************/
clear all
set more off

if "`c(username)'" == "tomas1608" { 
	global main "C:/Users/tomas1608/Maestria/Metodos/TPs-metodos/TP2/"
}
else if "`c(username)'" == "juli" { 
	global main ""
}

else if  "`c(username)'" == "Usuario" {
	global main "C:/Users/Usuario/Desktop/Trabajo - UdeSA/Maestría 2024/Cursada/Segundo trimestre/MÉTODOS ECONOMÉTRICOS Y ORGANIZACIÓN INDUSTRIAL APLICADA/TPs-metodos/TP2"
}

else if  "`c(username)'" == "tutor" {
	global main ""
} // Coloque aquí su dirección y nombre de usuario en "tutor"

global input "$main/input"
global output "$main/output"

cap mkdir "$main/output/tables"
cap mkdir "$main/output/figures"

use "$input/data_blp_loopf.dta", clear

************************************************************
*  					   Ejercico 4                          *
************************************************************

* Inciso b.
*=============
*En este apartado estimamos las elasticidades, propias y cruzadas, para la tienda 9 en la semana 10. Para esto, para poder comparar las elasticidades entre logit y BLP, será necesario modificar las estimaciones de Logit, puesto que BLP utiliza los market_share agregados a nivel de las 4 grandes marcas.
*Además, a diferencia de Logit del ejercicio 3, readaptamos la estimación para utilizar solo la variabilidad de las variable marca-semana como en BLP, tomando los promedio de las variables para todas las tiendas de una misma semana-marca. Este modelo, al igual que en el modelo de BLP utilizado, no incluye constante:
egen avg_precio = mean(precio), by(semana brand_id)
egen avg_descuento= mean(descuento), by(semana brand_id)
egen avg_delta = mean(delta_jpt_big), by(semana brand_id)
ivregress 2sls avg_delta (avg_precio=costo hausman_iv pricestore*) avg_descuento i.brand_id, nocons robust // Coeficiente: -0.8966512 


*Elasticidades para la semana 10 y tienda 9:
keep if tienda == 9 &  semana == 10

*Calculamos el market share para este caso:
total(ventas)
display(85/0.64) // Igual a 132.8125

*Calculamos el market share a nivel de marca para la tienda 9 y semana 10:
gen share_marca910= ventas/132.8125


*  ELASTICIDADES LOGIT   *
*Luego, generamos la matriz de elasticidades:
sort marca
matrix elasticidades_logit = J(11, 11, 0)

local coeficiente_elasticidad_logit = -0.8966512  // Derivado de la estimación previa

*Construimos el loop que genera las elasticidades tanto propias como cruzadas de los 11 productos:
forvalues i = 1/11 {
    forvalues j = 1/11 {
        if `i' == `j' {
            matrix elasticidades_logit[`i', `j'] = `coeficiente_elasticidad_logit' * precio[`i'] * (1 - share_marca910[`i'])
        } 
		else {
            matrix elasticidades_logit[`i', `j'] = -`coeficiente_elasticidad_logit' * precio[`j'] * share_marca910[`j']
        }
    }
}

matrix list elasticidades_logit, format(%9.3f)

*Exportamos la matriz a un archivo .txt:
capture log close
log using "$output/tables/elasticidades_logit.txt", text replace
matrix list elasticidades_logit, format(%9.3f)
log close


*	ELASTICIDADES BLP    *
*Luego, generamos la matriz de elasticidades:
sort marca
matrix elasticidades_blp = J(11, 11, 0)

local coeficiente_elasticidad_blp = -1.058827 // Derivado de BLP


*Construimos el loop que genera las elasticidades tanto propias como cruzadas de los 11 productos:
forvalues i = 1/11 {
    forvalues j = 1/11 {
        if `i' == `j' {
            matrix elasticidades_blp[`i', `j'] = `coeficiente_elasticidad_blp' * precio[`i'] * (1 - share_marca910[`i'])
        } 
		else {
            matrix elasticidades_blp[`i', `j'] = -`coeficiente_elasticidad_blp' * precio[`j'] * share_marca910[`j']
        }
    }
}

matrix list elasticidades_blp, format(%9.3f)

*Exportamos la matriz a un archivo .txt:
capture log close
log using "$output/tables/elasticidades_blp.txt", text replace
matrix list elasticidades_blp, format(%9.3f)
log close


* Inciso c.
*=============
*Ahora, estimamos los costos marginales para la tienda 9 en la semana 10 bajo el supuesto de que cada marca tiene un solo dueño. Por lo tanto, podemos utilizar la fórmula del Índice de Lerner para despejar el valor del costo marginal que dependerá de la elasticidad de cada marca y de los precios.
*Entonces, primero generamos la variable elasticidad que mantendrá como valor la elasticidad de BLP de cada una de las 11 marcas, según corresponda, independiente de la semana-tienda:
gen e_precio = . 
replace e_precio = -3.221 if marca == 1
replace e_precio = -4.380 if marca == 2
replace e_precio = -6.094 if marca == 3
replace e_precio = -2.816 if marca == 4
replace e_precio = -5.306 if marca == 5
replace e_precio = -8.817 if marca == 6
replace e_precio = -2.740 if marca == 7
replace e_precio = -3.403 if marca == 8
replace e_precio = -4.109 if marca == 9
replace e_precio = -1.641 if marca == 10
replace e_precio = -4.718 if marca == 11

*Luego, generamos la variable que corresponde al costo marginal de acuerdo al despeje matemático a partir del Índice de Lerner:
gen costo_mg_blp=((1/e_precio)+1)*precio

*Exportamos la tabla en formato .tex:
outsheet costo costo_mg using "$output/tables/resultados_costos.txt", replace


************************************************************
*  					   Ejercico 5                          *
************************************************************
* En este apartado predecimos los precios usando el modelo logit (sin coeficientes aleatorios) luego de la fusión pero solo para la semana 10. En este sentido, nos desvíamos de la consigna original, ya que el algoritmo de fusión para la semana 10 y tienda 9 no convergía al no econtrar diferencias significativas entre los precios de la pre-fusión y post-fusión. Por lo tanto, nos concentramos en la fusión con respecto a la semana 10, para poder comentar el ejercicio correctamente.

*Para esto, preparamos la base de datos siguiendo los condicionamientos necesarios para el código que realiza la simulación del precio para la fusión ("mergesim"). Entonces, necesitamos una variable que define el tamaño del mercado, es decir, la cantidad potencial de compradores por marca-semana:

use "$input/data_blp_loopf.dta", clear
sort marca semana
by marca semana: egen market_size = total(cantidad)

*Luego, seteamos el panel data a nivel marca-tienda, tomando solo la variabilidad en el tiempo:
egen marca_tienda = group(marca tienda)
xtset marca_tienda semana

*De acuerdo con la documentación de "mergesim", seteamos la configuración básica del modelo:
mergersim init, price(precio) quantity(ventas) marketsize(market_size) firm(marca)

*Realizamos la regresión con el panel data, incluyendo los efectos fijos temporales. Donde "M_ls", variable creada anteriormente en "mergersim init", corresponde al logaritmo del cociente entre el share estimado de la marca j y el sahre del bien externo:
xtreg M_ls precio descuento semana, fe robust

*Analizamos la condiciones pre-mercado para la semana 10 y tienda 9:
mergersim market if semana == 10 

*Por último, el tercer paso implica simular los precios para la fusión de las 3 grandes marcas. Por lo tanto, generamos una nueva variable que corresponde a la fusión de las 3 grandes marcas, es decir, incluye los primeros 9 productos:
gen fusion= 0
replace fusion = 1 if marca == 1
replace fusion = 1 if marca == 2
replace fusion = 1 if marca == 3
replace fusion = 1 if marca == 4
replace fusion = 1 if marca == 5
replace fusion = 1 if marca == 6
replace fusion = 1 if marca == 7
replace fusion = 1 if marca == 8
replace fusion = 1 if marca == 9

*Por último, exportamos los resultados de la fusión:
mergersim simulate if semana == 10, newfirm(fusion) detail
keep if semana == 10 & tienda == 9
outsheet precio M_price2 M_price_ch using "$output/tables/fusion.tex", replace
