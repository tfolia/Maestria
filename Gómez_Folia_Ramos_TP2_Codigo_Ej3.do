/*******************************************************************************
                          Universidad de San Andrés	         
                             Métodos Econométricos
							  Trabajo Práctico 2
								 Código Ej. 3
								  Julio 2024
					Facundo Gómez, Tomás Folia y Julian Ramos
********************************************************************************/
clear all
set more off

if "`c(username)'" == "tomas1608" { 
	global main "C:/Maestria/Metodos/TPs-metodos/TP2"
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

import excel using "$input/DATA_UDESA.xlsx", firstrow  clear

************************************************************
*  							LOGIT
************************************************************
*Contruimos los market share:
sort semana tienda marca

*Calculamos el total de ventas por semana:
bysort semana: egen sum_market_sale=total(ventas)

*Esto es el 64% del mercado total. Entonces, las ventas totales del mercado son:
gen tot_market_sale= sum_market_sale/0.64

*Calculamos el market share a nivel de semana-tienda-marca:
gen real_ms_per_brand_and_store= ventas/tot_market_sale
gen log_real_ms_per_brand_and_store=ln(real_ms_per_brand_and_store)

*Ahora, calculamos delta_jpt=ln(s_jpt_hat)-ln(market share del bien externo):
*Suponemos que el market share del bien externo es constante por tienda y semana, por lo que el market share semana-tienda del bien externo es 0.36:
gen delta_jpt=log_real_ms_per_brand_and_store-ln(0.36)

* Inciso 1.
*=============
*Realizamos MCO de la utilidad media de la marca j en la tienda p en la semana t contra el precio y promoción como las características de los productos:

reg delta_jpt precio descuento, robust
outreg2 using "$output/tables/tabla1.tex", replace label keep(precio descuento) ctitle(Modelo 1) addtext(Dummies por Marca, No,  Dummies por Tienda, No, Dummies por Marca-Tienda, No)


* Inciso 2.
*=============
*Realizamos MCO de la utilidad media de la marca j en la tienda p en la semana t
*contra el precio y promoción como las características de los productos, y 
*dummies por marca:
reg delta_jpt precio descuento i.marca, robust
outreg2 using "$output/tables/tabla1.tex", append label keep(precio descuento) ctitle(Modelo 2) addtext(Dummies por Marca, Si, Dummies por Tienda, No, Dummies por Marca-Tienda, No)


* Inciso 3.
*=============
*Realizamos MCO de la utilidad media de la marca j en la tienda p en la semana t
*contra el precio y promoción como las características de los productos. Además, se incluyen dummies por marca-tienda:
reg delta_jpt precio descuento i.marca##i.tienda, robust
outreg2 using "$output/tables/tabla1.tex", append label keep(precio descuento) ctitle(Modelo 3) addtext(Dummies por Marca, Si, Dummies por Tienda, Si, Dummies por Marca-Tienda, Si)


* Inciso 4.
*=============
*Realizamos 2SLS de la utilidad media de la marca j en la tienda p en la semana t contra el precio, que ha sido instrumentado por la variable de "costo", y promoción como las características de los productos. Se repiten los 3 modelos de los incisos anteriores, por lo que incluimos dummies por marca y por marca-tienda:
ivregress 2sls delta_jpt (precio=costo) descuento, robust
outreg2 using "$output/tables/tablaiv1.tex", replace label keep(precio descuento) ctitle(Modelo 3) addtext(Dummies por Marca, No, Dummies por Tienda, No, Dummies por Marca-Tienda, No)

ivregress 2sls delta_jpt (precio=costo) descuento i.marca, robust
outreg2 using "$output/tables/tablaiv1.tex", append label keep(precio descuento) ctitle(Modelo 4) addtext(Dummies por Marca, Si, Dummies por Tienda, No, Dummies por Marca-Tienda, No)

ivregress 2sls delta_jpt (precio=costo) descuento i.marca##i.tienda, robust
outreg2 using "$output/tables/tablaiv1.tex", append label keep(precio descuento) ctitle(Modelo 5) addtext(Dummies por Marca, Si, Dummies por Tienda, Si, Dummies por Marca-Tienda, Si)



* Inciso 5.
*=============
*Para utilizar la estrategia de Hausman, calculamos para una marca-semana particular el precio promedio en otros mercados, definiendo como "otros mercados" a otras "tiendas" de esa misma marca en la misma semana.
*Primero, creamos la variable que será el instrumento
gen hausman_iv = .

*Luego, obtenemos la lista de tiendas:
levelsof tienda, local(store)

local a = 1

*Por último, reemplazamos con el precio promedio de las tiendas restantes para una misma marca-semana.
forval marca = 1(1)11 {
    forval week = 1(1)48 {
        foreach negocio of local store {
            qui sum precio if tienda != `negocio' & semana == `week' & marca == `marca'
            qui replace hausman_iv = `r(mean)' if marca == `marca' & semana == `week' & tienda == `negocio'
            di `a'
            local a = `a' + 1
        }
    }
}

*Realizamos el IV:
ivregress 2sls delta_jpt (precio=hausman_iv) descuento, robust
outreg2 using "$output/tables/tablaiv2.tex", replace label keep(precio descuento) ctitle(Modelo 6) addtext(Dummies por Marca, No, Dummies por Tienda, No, Dummies por Marca-Tienda, No)

ivregress 2sls delta_jpt (precio=hausman_iv) descuento i.marca, robust
outreg2 using "$output/tables/tablaiv2.tex", append label keep(precio descuento) ctitle(Modelo 7) addtext(Dummies por Marca, Si, Dummies por Tienda, No, Dummies por Marca-Tienda, No)

ivregress 2sls delta_jpt (precio=hausman_iv) descuento i.marca##i.tienda, robust
outreg2 using "$output/tables/tablaiv2.tex", append label keep(precio descuento) ctitle(Modelo 8) addtext(Dummies por Marca, Si, Dummies por Tienda, Si, Dummies por Marca-Tienda, Si)


* Inciso 6.
*=============

*Elasticidad promedio de cada una de las marcas para cada resultado de regresión:
gen e_1=precio*(-0.0520378)*(1-real_ms_per_brand_and_store)
gen e_2=precio*(-0.3523115)*(1-real_ms_per_brand_and_store)
gen e_3=precio*(-0.3013762)*(1-real_ms_per_brand_and_store)

bysort marca: egen e_prom_1=mean(e_1)
bysort marca: egen e_prom_2=mean(e_2)
bysort marca: egen e_prom_3=mean(e_3)

*Exportamos la tabla:
preserve
collapse (mean) e_prom_1=e_1 e_prom_2=e_2 e_prom_3=e_3, by(marca)
outsheet marca e_prom_1 e_prom_2 e_prom_3 using "$output/tables/resultados_elasticidades.txt", replace
restore

save "$input/data_logit.dta", replace

* BASE DE DATOS BLP
*==========================
*Este breve apartado del código solo genera las variables asociadas con la distribuciones de ingreso aleatorias que serán necesarias para el algoritmo de BLP.
import excel using "$input/DATA_UDESA.xlsx", sheet("Variables demograficas") firstrow clear
drop E F G H I
save "$input/temp_sheet2.dta", replace

use "$input/data_logit.dta", clear
merge m:1 tienda using "$input/temp_sheet2.dta", nogen
drop cantmujeres educacion
drop if tienda==88

*Calculamos el desvío estándar del ingreso de todas las tiendas:
summarize ingreso
local desvio_ingreso = r(sd)

*Luego, generamos las 19 variables con distribuciones normales que tiene como media el ingreso de cada hogar o tienda y desvío estándar el desvío de los datos de ingreso de todas las tiendas:
set seed 123
forvalues i = 1/19 {
    gen income_dist`i' = rnormal(ingreso, `desvio_ingreso')
}

save "$input/data_blp_final.dta", replace
