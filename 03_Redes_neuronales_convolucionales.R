"CNN: Convolutional Neural Network. La de las casillitas"

# Kernel = Convolución
#Filters: lo normal no es aprender sólo una convolución, porque encontces detectare'un patrón- Si pongo N filters, se harán
# N convoluciones de 3x3, N patrones distintos. 
#una capa convolucional devuelve una imagen, no un numero. 
laver_conv_2d(kernel_size=c(3,3), activation="relu", filters = 32) 