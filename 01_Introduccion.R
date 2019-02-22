# R Tiene datasets precargados . Por ejemplo datos sobre los pasajeros del Titanic
library(tidyverse)
data("Titanic")
View(Titanic)

titanic <- as.data.frame(Titanic)

#------------Benchmark
#aquí se ponen los pesos porque hay una columna freq en Titanic para indicarme que ese dato se repitió N veces. 
# esto de los pesos se usa mucho para contrabalancear datasets desbalanceados (dataasets con muchos ceros y pocos unos)
# en esos casos se pesan mucho los 1's para que afecten
# vamos a usar una logística (porque es si y no)
benchmark <- glm(data=titanic,formula=Survived~Class+Sex+Age,family="binomial",weights = titanic$Freq)  # build linear regression model on full data

summary(benchmark)

# Vamos a ver la prección. Tengo que elegir dónde poner el punto de corte. 
# Dependerá de si quiero ser más sensible o más espefícico.
#
# type="response": en el espacio de respuesto, es decir , la y, queva de 0 a 1
# type="link": es el espacio de linkm que va en el eje de las X. 
predict(benchmark, newdata = titanic, type="response") > 0.6
prediccionBenchmark<- predict(benchmark, newdata = titanic, type="response") >0.6
# El AIC se usa mucho para comparar modelos. Es un un numerito que a bote pronto te dice lo bueno que es o no.
# No deberíamos poder comparar los AIC así como te apetezca.
# Lo0s científicos de datos poco rigurosos lo hacen, incluso en kaggle se ve. Pero un estadístico se llevaría con razón
# las manos a la cabeza

install.packages("e1071")
library(e1071)

#--- Primer clasificador: naiveBayes
# El teorema de Bayer: puedo aplicar una fórmula para poder ir actualizando las prb de que ocurran cosas.
# Cuando no tengo datos, todo una probabilidad a priori. Por ejemplo, si no tengo datos, puedo suponer que
# el 50 % de las persoinas van a ser supervivientes 
# Lección: ese a priori lo eliges. 
# Luego se van actualizando esas probabilidad es con probabilidades condicionadas. 
# Cual es la prob de que una persona sobreviva si es tripulación?
#F1 para ayuda

#OJO: Aquí usamos 
#este paquete no admite weights, pero tiene la cosa rara de que es capaz de interpretar los pesos si le pasas
#el dataset tal cual, que es una tabla de frecuencias.
# Este modelo da factores independientemente según cada variable. No posibilita combinación 
#mal
nb <- naiveBayes(Survived~Class+Sex+Age,data=titanic)

#BIEN
nb <- naiveBayes(Survived~Class+Sex+Age,data=Titanic)


"
-priori probabilities:
Survived
No      Yes 
0.676965 0.323035 . Esto es del todo dataset tal cual. 

Conditional probabilities: Aplicando bayes. 
Class
Survived        1st        2nd        3rd       Crew
No  0.08187919 0.11208054 0.35436242 0.45167785
Yes 0.28551336 0.16596343 0.25035162 0.29817159

Sex
Survived       Male     Female
No  0.91543624 0.08456376
Yes 0.51617440 0.48382560

Age
Survived      Child      Adult
No  0.03489933 0.96510067
Yes 0.08016878 0.91983122> 
"

summary(nb)

prediccionNb<- predict(nb, newdata = titanic)

View(cbind(titanic, prediccionBenchmark, prediccionNb))

#Mujeres de 3ª adultas tienen 89No y 76 Si. Está casi empataado, y un modelo se decanta por un lado
# y el otro por el otro

# Intuitivamente, si dice en los demás casi lo mismo es que mejora poco. 

# El problema es que hay interacciones entre las varaibles (no es lo mismo ser mujer de 1ª que de segunda)
# Esto se llama supuesto de aditividad. 

# Este dataset hay más aditividad de lo que esperábamos, y si no lo tenemos en cuenta estamos comentiendo errores
# epistémicos for falta de definición de modelo. 
# Y no hay nada que me lo diga, para eso están los benchmarks. 

# Aquí añadimos intracción entre Sexo y Clase
nb2 <- naiveBayes(Survived~Class+Sex+Age+Sex*Class, data=Titanic)
nb
nb2

# NAIVE BAYES NO PERMITE INTERACCIONES,POR ESO SALE LO MISMO

# Tengo una hipótesos: en el Titanic hay una interaccíon entre Sexo y Clase. Pero eso tengo que probarlo
# no puedo decir a priori que lo sale.
# Para ello, de nuevo, los benchmarks. Hago un contraste entre un modelo con interacciones y otro sin. 

#-------------------------- ARBOLES DE DECISIÓN ---
# Ver http://www.r2d3.us/una-introduccion-visual-al-machine-learning-1/
# Los árboles son muy flexibles: eso e bueno pero tb puede ser malo porque pueden tener al overfitting
# es decir,que si cambio los datos va y falla. 

# En qué orden eligjo las bifurcaciones.En principio, cojo un método greedy: la varaible que sé que más afecta 
# va a la primera bifurcación. Eso a veces dará problema. Otro forma es elegir la variable que más divida. 
# Pero no siempre es mejor esto. No siempre es la mejor estrategia empezar por el movimiento más ambicioso,
# que es el mov  que come más piezas al adversario. 
#C audno acabo: hay varios criterios:
# - Si ya tengo el 100% de datos de un tipo, no tiene sentido hacer más bifuraciones.
# - Limitar la profundidad.
# - Elegir el umbral del primero punto en otro valor, por ejemplo en el 95%. 

#Otro punto a favor es que es fácil de interpretar. 

# El árbol no sólo clasifica entre dos opciones, sino entre varias.

# Principal problema: el posible overfitting. 

#Este modelo es muy poco lineal. Entonces no hay ténicas como el gradiente (moverse un poquito hasta que encuentro el mejor)

# OJO; Lineal = un pequeño cambio en el input tendrá un cambio proporcional en la salida. 
# No quiere decir ques ea lineal.Porque una logística tb es lineal. 

#Otra ventaja es que el arbol no tienes que decirle cuales son las interacciones, él se da cuenta.

# El modleo de árbol no te garantiza que sea el mejor árbol . Es es lo normal en ML. 
# Sin embargo, GLM o LM te da el mejor dentro de esa familida, porque se calcula matemáticamente, no iterativamente
## como el árbol y otros muchos. 

# Hiperparámetro: igual que en un model tienes las betas. El hiperparámetro no es del modelo, sino del método
# que genera el modelo: por ejemplo, la máxima profundidad de árbol, o cómo eliges la primera variable. 
# o decir cual es el criterior de correlación (Pearoson Speechman, etc). La familia tb podría ser un hiperparámetro 
# (decir que es ax + b)
#install.packages("tree")
library(tree)
library(dplyr)
library(randomForest)
#install.packages("titanic")
library(titanic)
library(caret)
#install.packages("ROCR")
library(ROCR)

titanic_train # datos de entrenamiento de titanic
tree(Survived ~ Age, titanic_train) # esto no es normal, no voy a hacer un árbol de una sola división 

" split, n, deviance, yval (valor estimado por el árbol)
1) root 714 172.20 0.4062 (40% sobrevive) 
  2) Age < 6.5 47 (número de supervivientes)   9.83   0.7021 (70% de los pequeños vive) *  
  3) Age > 6.5 667(número de superviientes)    158.00 0.3853 (38 % de los grandes vive) *
"
arbol <- tree(Survived ~ Age + Pclass, titanic_train)

arbol
"
1) root 714 172.200 0.4062  
   2) Pclass < 2.5 359  87.940 0.5710   Primera y segunda clase
     4) Age < 17.5 35   2.743 0.9143 * (El asterisco dice que es nodo hoja. Aquí para porque ya tienes un 91% de supervivencia)
     5) Age > 17.5 324  80.630 0.5340  
      10) Pclass < 1.5 174  40.190 0.6379   Se pueden reutilizar variables. 
        20) Age < 44.5 107  20.670 0.7383 *
        21) Age > 44.5 67  16.720 0.4776 *
      11) Pclass > 1.5 150  36.370 0.4133 *
   3) Pclass > 2.5 355  64.650 0.2394  Tercera clase
     6) Age < 6.5 30   7.367 0.5667 *
     7) Age > 6.5 325  53.770 0.2092 *
"
predict(arbol,newdata=titanic_test)
"Hay muchos valores que se repiten,porque un árbol encasilla en valores, no es una regresión. "
# El árbollo que te da al finalk son probabilidades de que sobreviva o no. GTú endrás que elegir el 
# umbral viendo la ROC. 
# Si me hago una mtriz de coste (poner qué me cuestan metafoŕicamente los falsos positivos, los falsos 
# negativos etc. )
# si tengo esto + la roc podré elegir perfectamente el umbral. 

# RANDOM FOREST----

titanic_train <- titanic_train[!is.na(titanic_train$Age),] # r plano
titanic_train <- titanic_train %>% filter(!is.na(Age)) %>%  mutate(Survived=factor(Survived))

library("randomForest")
#Habría que convertir Survived a un factor para que no haga regresión, sino clasificacíon. 
randomForest(Survived ~ Pclass + Age, titanic_train)

# Se calculan varios árboles (unos cogen unas variables, otros empiezan por otra variale, etc).
# Cada árbol vota, y se hace por democracia. (para un dato, uno dice 0.6 de prob, otro 0.7 de priob de sobrevivir
# y se hace la media) 
#Mejor accuracy. Pero recuerda que maximizar acuracy no siempre es lo mejor
# Random forest pierde explicabilidad
# tb puede hacer regresión o clasificación. 

# KNN ------

library(class)

# ¿Cual es el defecto de un árbol de decisión?
# Sólo puede cortar en rectángulos. 

# KNN K-Vecinos-Cercanos
# K es un hiperparámetro
# Para un punto nuevo, se busca el punto más cercano del conjunto de entrenamientoy se le da ese valor. 
#Hay varias formas de medir distancias
# - Euclidea: para toda la vidad
# - Coseno: para datasets con variables con mucha varianza (todo está muy lejo, por ejemplo en textos)
#    , funciona mejor. Mide la distancia en ángulo. En textos cuento las palabras. No me interesa cuantas palabras tengas
#sino si todas van de lo mismo. 
# otro ejemplo: documentos de textos de nuevo. Tengo un tweet muy corto de tecnología, tengo en las X el número de veces de 
# la palabra PC, y en la Y el número de la palabra arroz. 
# Si tengo después un libro que dice 10000000 veces la palabra PC yuna vez la paabraarrox estárá a mucha distancia euclidea
# del tweet. Sin embargo, el coseno será muy parecido. Por eso me conviene. 

# Para el titanic:

" Puede tener sentido? "

" Librería class hace un modelo de knn"

"a veces poner un k muy alto. Si todos los puntos están muy pegados, da igual 1 que 50."
" sin embargo, si el dataset está muy disperso, la k no debería ser muy grande porque pillaré cosas de otra categoría. "

"diferencia con k mins: k mins no pregunta a los vecinos, simplemente busca centros (agrupar) y no es supervisado."

"Para el titanic no nos va a ir muy bien, porque hay mucha interacción entre variables => los vecinos están en
una 'línea' marcada por dos varia les, y el knn crece esféricamente"

"El problema de poner dos varaibles que correlaciones mucho no va a empeorar el modelo, pero si me va a imedir
explicarlo bien"

"Uno bueno será el 'iris'. Flores con tamaño te pétale y sépalo parecido tendrá una especie parecida"

ggplot(iris) + geom_point(aes_string(x="Petal.Length", y="Petal.Width", color="Species"))

#está muy agrupados por color, perfecto para knn

nrow(iris)

#divido entre entrenamiento y test
#Dos formas:
# 1)
sample(1:nrow(iris), 110)
# 2)
iris %>% sample_n(110)

idxTraining <- sample(1:nrow(iris), 110)
iris_training <- iris[idxTraining,]
iris_test <- iris[-idxTraining,]

# Regla dpara elegir el training: El 80-20 es mentira
# hay técnicas que requiere nmuchos datos para entrenar. Pues las que sean necesarias
# en el test no tiene por qué ir el resto, sino lo mínimo necesario par estimar bien el error.

#En red neuranal suele ser 99 para entrenar , 1 para probar.
# En otros otra cosa. 

#Esta knn te da las variables de training, las variables de test y un ventos de lo que quiero consedguir en training.

#Resultado correcto
train_output = iris_training$Species

#Quitamos columna de especia de ambos datasets (porque si no, las meterá en el modelo y saldrá perfecto.)
#iris_training <- iris_training[,-5]
colnames(iris_training)
iris_training <- iris_training[, colnames(iris_training) != "Species"]
iris_test <- iris_test[, colnames(iris_test) != "Species"]

#más bonito, con deplyr:
# iris_training %>% select(-Species)

knn(iris_training, iris_test, train_output)

# k = 1. Un vecino. prob: T. Atributo de probabilidad  indicando que el atributo pertenesca a esa clase. 
knn(iris_training, iris_test, train_output, k = 1, prob=T)

knn(iris_training, iris_test, train_output, k = 5, prob=T)

knn(iris_training, iris_test, train_output, k = 80, prob=T) # está mal, cojo todo el dataset para "votar"

#El prob indica si el % de los vecinoshan coincidido enla decisión. 