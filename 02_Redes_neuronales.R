"
En una imagen la X de mi problema no son la X e Y de la imagen.
Los datos son las intensidades.
"

"Para detectar el númoer que hay escrito me sirve de pongo saber que hay una intensidad ed negro 0.8 en el pixel 
de columna 7 y 8 

Estos datos se sacan del servicio postal de EEUU, dode encontrar el código postal te ayuda mucho a ya redirigr la carta automáticamente
"
install.packages("keras")
library("keras")
install_keras()

mnist <- dataset_mnist()
str(mnist) # ya tiene un training (60000 imágenes de 28x28) y test (10000 imaágenes de 28x28)

image(as.matrix(mnist$train$x[2,,]))


"Transformacmos un poco el datset"

"La matriz de entrada no puede ser tridimensional, tengo la dimensión de número la divisíon de largo yla de ancho"
"lo aplano=flatten"

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

" es decir, tengo una rejilla de 28 x 28 por cada numero, "
" y quiero llegar a tener 28x28 columnas por cada numero, para que sea ya un dataframe bidimensional "
# devtools::install_github("rstudio/tensorflow")
dim(x_train) <- c(60000,784)
dim(x_test) <- c(10000,784)

#normalizar el input (de 0 a 255) a (0.1)
x_train <- x_train/255
x_test <- x_test/255

# one-hot encoding o dummay variable.
# convierto la variable valor, que será 5, 6 etc, a 10 columnas del 0 al 9 sacando un 1 si es es esenúmero. 
# así lo entiende mejor.
y_train <- to_categorical(y_train,10)
y_test <- to_categorical(y_test,10)

# Modelo 
modelo <- keras_model_sequential()  # red neuronal secuencial las normales que hemos visto, las primeras
modelo  %>% 
  #Capa totalmente conectada, densa, con 200 neuronas, cada una con 784 entradas (28x28) y con un umbral tipo relu  
  layer_dense(unit=200, activation = "relu", input_shape = c(784), name = "Nombre_que_puedo_elegir")  %>% 
  layer_dropout(rate = .4 ) %>%  # ponemos aleatoriamente a 0 algunas neuronas para evitar overfitting. "Cuando le haces putadas a las redes, mejoran"
  layer_dense(unit=100, activation = "relu") %>% 
  layer_dropout(rate = .3 ) %>% 
  layer_dense(unit=10, activation = "softmax") # capa de salida. La activación se usa softmax que te normaliza a 1. Se usa mucho en las de salidaEn este caso mi odelo es de dos capaz, porque lo de salida no cuenta

summary(modelo)

" las capas de pueden congelar: aprenderla y usarla posteriormente, por ejemplo para coger una red entrenada por otro"

# Función de coste . Ya heos hecho en otras clases, el RMSE es otra. Es una función para evaluar el modelo. Si esa función da un valor bajo, es que es bueno
# En RN tenemos la cross entropy y otras. 
# También se pueden pesar los casos , por ejenplo, decir que acertar los 0 son más importantes que los 8s. 
# Cuidado coj lo que se pide en una funcion deocste, porque puede ir en nuestra contra. 

modelo %>% 
  compile(loss="categorical_crossentropy",
          optimizer=optimizer_rmsprop(),
          metric=c("accuracy")) # metrica paraver si está aprendiendo. Es muy típico que se tire 4 días entranando y al 5 empiece a resolver el problema
#Es una sintaxis rara, porque lo haes pero no se guarda en una variavle

resultado <- fit(modelo, x_train, y_train, epochs = 50,  #epochs = número de itreaciones
                 batch_size = 128 , validation_split = 0.2) # las redes funcionan mejor no dandole todos los datos, sino dandlo trozos
                                # distitnos en cada epoch. En este caso vamos a meter sólo 128 dígitos en cada epoch. Poner potencias de 2 
                                # el validation_split es para ver qué porcentaje del dataset calcula el accuracy por cuestiones de computación. 
                                # no tiene nada que ver con  train y test
# Si tengo batch size muy pequeño, tendré que dar muchas epochs. En general, mejor un batch_size alto ante la duda. 

#Optimizer, son cómo va a a hacer para apernder, por ejemplo, el learning reate. 
# -hay gente que empieza con un LR alto y después si ve que salta , lo va bajando
# Optimizer famosoos:
# optimizer_adam
# optimizer_sgd => Stotastic Gradient Descent
# optimizer_rmsprop => 

"  4s 79us/step - loss: 0.0569 - acc: 0.9855 - val_loss: 0.1312 - val_acc: 0.9785 "
"acc es el accuracy más alto que ha econtrado durante el procesp"
"val_acc es el accuracy  final. A veces sl accuracy pede bajar como auímás alto que ha econtrado durante el procesp"
plot(resultado)
"OJO, este training y validation es del 0.2 que le hemos puesto a kears. "
" si la red la dejas funcionando más tiempo mas allá de tu objetivo llegarás a overfitting, así que no lo hagas. "

"muchas veces se divide en tres. Uno para entrenar, otro para validacíon y otro que no uso ahasta el final del todo para validar"
" ese tercer dataset es un mecanismo de proteccíon de overfitting. Porque cuando vas cambiando muchas veces lops hiperparámetros para probar
estás haciendo un overfitting de metaparámetros. Se pueden llamar train test validation, por ejemplo, aunque hay de todo"

modelo %>% evaluate(x_test,y_test) # evaluamos con datos de test. Usará la  métrica que le pasé antes. 

modelo %>% predict_classes(x_test)

resultados_y = mnist$test$y
#loss es es el cross entropy. Es la función de coste. Difícil de esplicar el cross entrpy. 

# Ejercicio------
# Vamos a cambiar la arquitectura de la red . Vamos a poner capas convolucionales. 
errores <- ifelse(modelo %>% predict_classes(x_test) - resultados_y != 0, 1, 0)
sum(errores)
sitios_donde_ha_habido_fallo <- mnist$train$x[errores,,]
sitios_donde_ha_habido_fallo 
dim(sitios_donde_ha_habido_fallo)
image(as.matrix(sitios_donde_ha_habido_fallo[0,,]))

#Su solucíon, mejor desde luego
predicciones <- modelo %>%  predict_classes(x_test)
idxErrores <- which(predicciones != mnist$test$y) # falla la 116, la 150, la 248
image(as.matrix(mnist$train$x[248,,]))

# Matriz de confusiones. 
table(mnist$test$y,predicciones)
