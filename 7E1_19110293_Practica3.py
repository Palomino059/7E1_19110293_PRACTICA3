import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cv2
import math

import numpy as np
import skimage
from skimage import io



Img1 = cv2.imread('Img1.png',1)
Img2 = cv2.imread('Img2.png',1)

color = ('b','g','r')

res1 = cv2.resize(Img1, dsize=(280, 280))
res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2RGB)

res2 = cv2.resize(Img2, dsize=(280, 280))
res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2RGB)

################################## SUMA ####################################
############################### METODOLOGIA 1 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- SUMA -------------------------------------|#

#//////SUMA DE IMAGENES\\\\\\\\\\\\\\
sum_imag = cv2.add(res1,res2)

ax[0, 1].imshow(sum_imag)
ax[0, 1].set_title('SUMA DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Suma sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_sum = cv2.calcHist([sum_imag], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_sum, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma Suma Img')
ax[1, 1].axis('off')

#///////////Ecualizacion de Suma de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_sum = cv2.cvtColor(sum_imag,cv2.COLOR_BGR2YUV)
Ecua_sum[:,:,0] = cv2.equalizeHist(Ecua_sum[:,:,0])
Ecualizacion_suma = cv2.cvtColor(Ecua_sum,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_suma)
ax[2, 1].set_title('Suma de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Suma con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_suma = cv2.calcHist([Ecualizacion_suma], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_suma, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Suma Img Ecua')
ax[3, 1].axis('off')
plt.show()

################################ SUMA  #####################################
############################# METODOLOGIA 2 ################################
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- SUMA 2 -------------------------------------|#

#//////SUMA 2 DE IMAGENES\\\\\\\\\\\\\\

sum2_imag = res1 + res2

ax[0, 1].imshow(sum2_imag)
ax[0, 1].set_title('SUMA 2 DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Suma 2 sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_sum = cv2.calcHist([sum2_imag], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_sum, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma Suma2 Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de Suma 2 de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_sum = cv2.cvtColor(sum2_imag,cv2.COLOR_BGR2YUV)
Ecua_sum[:,:,0] = cv2.equalizeHist(Ecua_sum[:,:,0])
Ecualizacion_suma = cv2.cvtColor(Ecua_sum,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_suma)
ax[2, 1].set_title('Suma 2 de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Suma 2 con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_suma = cv2.calcHist([Ecualizacion_suma], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_suma, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Suma 2 Imgenes Ecualizadas')
ax[3, 1].axis('off')
plt.show()

################################ SUMA  #####################################
############################# METODOLOGIA 3 ################################
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- SUMA 3 -------------------------------------|#

#///////////////////SUMA 3 DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

sum3_imag = cv2.addWeighted(res1,0.5,res2,0.5,0)

ax[0, 1].imshow(sum3_imag)
ax[0, 1].set_title('SUMA 3 DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Suma 3 sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_sum = cv2.calcHist([sum3_imag], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_sum, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma Suma 3 Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de Suma 3 de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_sum = cv2.cvtColor(sum3_imag,cv2.COLOR_BGR2YUV)
Ecua_sum[:,:,0] = cv2.equalizeHist(Ecua_sum[:,:,0])
Ecualizacion_suma = cv2.cvtColor(Ecua_sum,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_suma)
ax[2, 1].set_title('Suma 3 de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Suma 3 con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_suma = cv2.calcHist([Ecualizacion_suma], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_suma, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Suma 3 Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

############################################################################
################################ RESTA  #####################################
############################ METODOLOGIA 1 ################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- RESTA -------------------------------------|#

#///////////////////RESTA DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

resta_imag = cv2.subtract(res1,res2)

ax[0, 1].imshow(resta_imag)
ax[0, 1].set_title('RESTA DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Resta sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_rest = cv2.calcHist([resta_imag], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_rest, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma resta Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de resta de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_rest = cv2.cvtColor(resta_imag,cv2.COLOR_BGR2YUV)
Ecua_rest[:,:,0] = cv2.equalizeHist(Ecua_rest[:,:,0])
Ecualizacion_resta = cv2.cvtColor(Ecua_rest,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_resta)
ax[2, 1].set_title('Resta de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de resta con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_resta = cv2.calcHist([Ecualizacion_resta], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_resta, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma resta Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

################################ RESTA  #####################################
############################ METODOLOGIA 2 ################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- RESTA 2 -------------------------------------|#

#///////////////////RESTA DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

Sustraccion = cv2.subtract(res1,res2)

ax[0, 1].imshow(Sustraccion)
ax[0, 1].set_title('RESTA DE IMAGENES 2 ')
ax[0, 1].axis('off')


#////////////Histograma de Resta sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_Sustraccion = cv2.calcHist([Sustraccion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Sustraccion, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma resta Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de resta de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_Sustraccion = cv2.cvtColor(Sustraccion,cv2.COLOR_BGR2YUV)
Ecua_Sustraccion[:,:,0] = cv2.equalizeHist(Ecua_Sustraccion[:,:,0])
Ecualizacion_Sustraccion = cv2.cvtColor(Ecua_Sustraccion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Sustraccion)
ax[2, 1].set_title('Resta de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de resta con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_Sustraccion = cv2.calcHist([Ecualizacion_Sustraccion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_Sustraccion, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma resta Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

################################ RESTA  #####################################
############################ METODOLOGIA 3 ################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- RESTA 3 -------------------------------------|#

#///////////////////RESTA DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

Absdiff = cv2.absdiff(res1,res2)

ax[0, 1].imshow(Absdiff)
ax[0, 1].set_title('RESTA DE IMAGENES 3 ')
ax[0, 1].axis('off')


#////////////Histograma de Resta sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_Absdiff  = cv2.calcHist([Absdiff], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Absdiff , color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma resta Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de resta de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_Absdiff = cv2.cvtColor(Absdiff,cv2.COLOR_BGR2YUV)
Ecua_Absdiff [:,:,0] = cv2.equalizeHist(Ecua_Absdiff [:,:,0])
Ecualizacion_Absdiff  = cv2.cvtColor(Ecua_Absdiff ,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Absdiff)
ax[2, 1].set_title('Resta de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de resta con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_Absdiff  = cv2.calcHist([Ecualizacion_Absdiff ], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_Absdiff , color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma resta Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

################################################################################
################################ DIVISION  #####################################
############################## METODOLOGIA 1 ###################################


Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- DIVISION -------------------------------------|#

#///////////////////DIVION DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

division_imag = cv2.divide(res1, res2)

ax[0, 1].imshow(division_imag)
ax[0, 1].set_title('DIVISION DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Division sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_divion = cv2.calcHist([division_imag], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_divion, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma divion Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de division de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_div = cv2.cvtColor(division_imag,cv2.COLOR_BGR2YUV)
Ecua_div[:,:,0] = cv2.equalizeHist(Ecua_div[:,:,0])
Ecualizacion_div = cv2.cvtColor(Ecua_div,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_div)
ax[2, 1].set_title('Division de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de division con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_div = cv2.calcHist([Ecualizacion_div], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_div, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Division Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

################################################################################
################################ DIVISION  #####################################
############################## METODOLOGIA 2 ###################################


Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- DIVISION 2 -------------------------------------|#

#///////////////////DIVION DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

Divide = cv2.divide(res1,res2)

ax[0, 1].imshow(Divide)
ax[0, 1].set_title('DIVISION DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Division sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_Divide = cv2.calcHist([Divide], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Divide, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma divion Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de division de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_Divide = cv2.cvtColor(Divide,cv2.COLOR_BGR2YUV)
Ecua_Divide[:,:,0] = cv2.equalizeHist(Ecua_Divide[:,:,0])
Ecualizacion_Divide = cv2.cvtColor(Ecua_Divide,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Divide)
ax[2, 1].set_title('Division de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de division con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_Divide = cv2.calcHist([Ecualizacion_Divide], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_Divide, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Division Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

#################################################################################
################################ MULTIPLICACION #################################
################################# METODOLOGIA 1 ###################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- MULTIPLICACION -------------------------------------|#

#///////////////////MULTIPLICACION DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

multiplicacion_imag = cv2.multiply(res1, res2)

ax[0, 1].imshow(multiplicacion_imag)
ax[0, 1].set_title('MULTIPLICACION DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma de Multiplicacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_mult = cv2.calcHist([multiplicacion_imag], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_mult, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma Multiplicacion Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de Multiplicacion de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_mul = cv2.cvtColor(multiplicacion_imag,cv2.COLOR_BGR2YUV)
Ecua_mul[:,:,0] = cv2.equalizeHist(Ecua_mul[:,:,0])
Ecualizacion_mult = cv2.cvtColor(Ecua_mul,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_mult)
ax[2, 1].set_title('Multiplicacion de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de division con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_mul = cv2.calcHist([Ecualizacion_mult], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_mul, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Multiplicacion Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

################################ MULTIPLICACION #################################
################################# METODOLOGIA 2 ###################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|-------------------------------- MULTIPLICACION 2 -------------------------------------|#

#///////////////////MULTIPLICACION DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

Multiply = cv2.multiply(res1,res2)

ax[0, 1].imshow(Multiply)
ax[0, 1].set_title('MULTIPLICACION DE IMAGENES 2')
ax[0, 1].axis('off')


#////////////Histograma de Multiplicacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_Multiply = cv2.calcHist([Multiply], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Multiply, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma Multiplicacion Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de Multiplicacion de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_Multiply = cv2.cvtColor(Multiply,cv2.COLOR_BGR2YUV)
Ecua_Multiply[:,:,0] = cv2.equalizeHist(Ecua_Multiply[:,:,0])
Ecualizacion_Multiply = cv2.cvtColor(Ecua_Multiply,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Multiply)
ax[2, 1].set_title('Multiplicacion de Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de division con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_Multiply = cv2.calcHist([Ecualizacion_Multiply], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_Multiply, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Multiplicacion Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

####################################################################################
################################ LOGARITMO NATURAL #####################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|------------------------------ LOGARITMO NATURAL 1 -------------------------------------|#

#///////////////////LOGARITMO DE IMAGENES\\\\\\\\\\\\\\\\\\\\\

c = 255 / np.log(1 + np.max(res1)) 
log_image = c *(np.log(res1 + 1)) 
log_image = np.array(log_image, dtype = np.uint8) 

ax[0, 1].imshow(log_image)
ax[0, 1].set_title('LOGARITMO DE IMAGENES ')
ax[0, 1].axis('off')


#////////////Histograma del Logaritmo Natural sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_log = cv2.calcHist([log_image], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_log, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma del Log Natural Imgenes')
ax[1, 1].axis('off')

#///////////Ecualizacion de Logaritmo Natural de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_log = cv2.cvtColor(log_image,cv2.COLOR_BGR2YUV)
Ecua_log[:,:,0] = cv2.equalizeHist(Ecua_log[:,:,0])
Ecualizacion_log = cv2.cvtColor(Ecua_log,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_log)
ax[2, 1].set_title('Logaritmo Nat en Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Log Nat con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_log = cv2.calcHist([Ecualizacion_log], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_log, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Logaritmo Natural Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

################################ LOGARITMO NATURAL 2#####################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|------------------------------ LOGARITMO NATURAL 2 -------------------------------------|#

#///////////////////LOGARITMO DE IMAGENE 2\\\\\\\\\\\\\\\\\\\\\

c = 255 / np.log(1 + np.max(res2)) 
log_image2 = c * (np.log(res2 + 1)) 
log_image2 = np.array(log_image, dtype = np.uint8) 

ax[0, 1].imshow(log_image2)
ax[0, 1].set_title('LOGARITMO DE IMAGEN 2 ')
ax[0, 1].axis('off')


#////////////Histograma del Logaritmo Natural 2 sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_log2 = cv2.calcHist([log_image2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_log2, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma del Log Natural 2 Imgen')
ax[1, 1].axis('off')

#///////////Ecualizacion de Logaritmo Natural de Imagenes\\\\\\\\\\\\\\\\\\\\\\
Ecua_log2 = cv2.cvtColor(log_image2,cv2.COLOR_BGR2YUV)
Ecua_log2[:,:,0] = cv2.equalizeHist(Ecua_log2[:,:,0])
Ecualizacion_log2 = cv2.cvtColor(Ecua_log2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_log2)
ax[2, 1].set_title('Logaritmo Nat en Imagenes Ecualizadas')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Log Nat con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_log2 = cv2.calcHist([Ecualizacion_log2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_log2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Logaritmo Natural Imgenes Ecualizadas')
ax[3, 1].axis('off')

plt.show()

#################################################################################
################################ RAIZ CUADRADA #####################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\
for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\

Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])

ax[3, 2].set_title('Histograma con Ecualizada')   
ax[3, 2].axis('off')


#|------------------------------ RAIZ CUADRADA -------------------------------------|#

#///////////////////RAIZ CUADRADA\\\\\\\\\\\\\\\\\\\\\
Raiz = (res1**(0.5))
Raiz_m = np.float32(Raiz)
cv2.imwrite('raiz.jpg',Raiz)
Raiz_g = cv2.imread('raiz.jpg',0)

#Imagen RAIZ
ax[0, 1].imshow(Raiz)
ax[0, 1].set_title('Raiz')
ax[0, 1].axis('off')


#Histograma Imagen RAIZ sin ecualizar

for i, c in enumerate(color):
    hist_Raiz = cv2.calcHist([Raiz_m], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Raiz, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Raiz')
ax[1, 1].axis('off')


#Ecualizacion imagen RAIZ
Ecua_Raiz = cv2.equalizeHist(Raiz_g)
#Ecualizacion_Raiz = cv2.cvtColor(Ecua_Raiz,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecua_Raiz)
ax[2, 1].set_title('Imagen Raiz Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen RAIZ ecualizada

for i, c in enumerate(color):
    hist_ecua_Raiz = cv2.calcHist([Ecua_Raiz], [0], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Raiz, color = 'gray')
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Raiz Ecua')
ax[3, 1].axis('off')
plt.show()


################################ RAIZ CUADRADA 2 #####################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])

ax[3, 2].set_title('Histograma con Ecualizada')   
ax[3, 2].axis('off')


#|------------------------------ RAIZ CUADRADA 2-------------------------------------|#

#///////////////////RAIZ CUADRADA 2\\\\\\\\\\\\\\\\\\\\\

Raiz = (res2**(0.5))
Raiz_m = np.float32(Raiz)
cv2.imwrite('raiz.jpg',Raiz)
Raiz_g = cv2.imread('raiz.jpg',0)

#Imagen RAIZ
ax[0, 1].imshow(Raiz)
ax[0, 1].set_title('RAIZ CUADRADA 2')
ax[0, 1].axis('off')


#Histograma Imagen RAIZ sin ecualizar

for i, c in enumerate(color):
    hist_Raiz = cv2.calcHist([Raiz_m], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Raiz, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Raiz')
ax[1, 1].axis('off')


#Ecualizacion imagen RAIZ
Ecua_Raiz = cv2.equalizeHist(Raiz_g)
#Ecualizacion_Raiz = cv2.cvtColor(Ecua_Raiz,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecua_Raiz)
ax[2, 1].set_title('Imagen Raiz Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen RAIZ ecualizada

for i, c in enumerate(color):
    hist_ecua_Raiz = cv2.calcHist([Ecua_Raiz], [0], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Raiz, color = 'gray')
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Raiz Ecua')
ax[3, 1].axis('off')
plt.show()

###############################################################################
################################ DERIVADA #####################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|------------------------------ DERIVADA -------------------------------------|#

#///////////////////IMAGEN DE DERIVADA\\\\\\\\\\\\\\\\\\\\\

Derivada = cv2.Laplacian(res1,cv2.CV_32F)
Derivada_m = np.float32(Derivada)
cv2.imwrite('Derivada.jpg',Derivada)
Derivada_g = cv2.imread('Derivada.jpg',0)

#Imagen DERIVADA
ax[0, 1].imshow(Derivada)
ax[0, 1].set_title('Derivada')
ax[0, 1].axis('off')


#Histograma Imagen DERIVADA sin ecualizar

for i, c in enumerate(color):
    hist_Derivada = cv2.calcHist([Derivada_m], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Derivada, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Derivada')
ax[1, 1].axis('off')


#Ecualizacion imagen DERIVADA
Ecua_Derivada = cv2.equalizeHist(Derivada_g)
ax[2, 1].imshow(Ecua_Derivada)
ax[2, 1].set_title('Imagen Derivada Ecua')
ax[2, 1].axis('off')
cv2.waitKey(0)


#Histograma Imagen DERIVADA ecualizada

for i, c in enumerate(color):
    hist_ecua_Derivada = cv2.calcHist([Ecua_Derivada], [0], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Derivada, color = 'gray')
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Derivada Ecua')
ax[3, 1].axis('off')

plt.show()

################################ DERIVADA 2 #####################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)
#|----------------------------- IMAGEN 1 ----------------------------------|#

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#|------------------------------ DERIVADA 2 -------------------------------------|#

#///////////////////IMAGEN DE DERIVADA 2\\\\\\\\\\\\\\\\\\\\\

Derivada2 = cv2.Laplacian(res2,cv2.CV_32F)
Derivada_m2 = np.float32(Derivada2)
cv2.imwrite('Derivada.jpg',Derivada2)
Derivada_g2 = cv2.imread('Derivada.jpg',0)

#Imagen DERIVADA
ax[0, 1].imshow(Derivada2)
ax[0, 1].set_title('Derivada')
ax[0, 1].axis('off')


#Histograma Imagen DERIVADA sin ecualizar

for i, c in enumerate(color):
    hist_Derivada2 = cv2.calcHist([Derivada_m2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Derivada2, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Derivada')
ax[1, 1].axis('off')


#Ecualizacion imagen DERIVADA
Ecua_Derivada2 = cv2.equalizeHist(Derivada_g2)
ax[2, 1].imshow(Ecua_Derivada2)
ax[2, 1].set_title('Imagen Derivada Ecua')
ax[2, 1].axis('off')
cv2.waitKey(0)


#Histograma Imagen DERIVADA ecualizada

for i, c in enumerate(color):
    hist_ecua_Derivada2 = cv2.calcHist([Ecua_Derivada2], [0], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Derivada2, color = 'gray')
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Derivada Ecua')
ax[3, 1].axis('off')

plt.show()

###############################################################################
############################### POTENCIA ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- POTENCIA -------------------------------------|#

#//////POTENCIA\\\\\\\\\\\\\\

potencia = np.zeros(res1.shape,res1.dtype)
g = 0.5
c = 1
potencia = c * np.power(res1,g)
maxil = np.amax(potencia)
potencia = np.uint8(potencia/maxil * 255)

ax[0, 1].imshow(potencia)
ax[0, 1].set_title('POTENCIA')
ax[0, 1].axis('off')


#////////////Histograma de Potencia sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_pot = cv2.calcHist([potencia], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_pot, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Potencia')
ax[1, 1].axis('off')

#///////////Ecualizacion de Potencia\\\\\\\\\\\\\\\\\\\\\\
Ecua_pot = cv2.cvtColor(potencia,cv2.COLOR_BGR2YUV)
Ecua_pot[:,:,0] = cv2.equalizeHist(Ecua_pot[:,:,0])
Ecualizacion_pot = cv2.cvtColor(Ecua_pot,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_pot)
ax[2, 1].set_title('Potencia Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Potencia con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_pot = cv2.calcHist([Ecualizacion_pot], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_pot, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Potencia Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### POTENCIA 2 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- POTENCIA 2-------------------------------------|#

#//////POTENCIA\\\\\\\\\\\\\\

potencia = np.zeros(res2.shape,res1.dtype)
g = 0.5
c = 1
potencia = c * np.power(res2,g)
maxil = np.amax(potencia)
potencia2 = np.uint8(potencia/maxil * 255)

ax[0, 1].imshow(potencia2)
ax[0, 1].set_title('POTENCIA 2')
ax[0, 1].axis('off')


#////////////Histograma de Potencia sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_pot2 = cv2.calcHist([potencia2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_pot2, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Potencia')
ax[1, 1].axis('off')

#///////////Ecualizacion de Potencia\\\\\\\\\\\\\\\\\\\\\\
Ecua_pot2 = cv2.cvtColor(potencia2,cv2.COLOR_BGR2YUV)
Ecua_pot2[:,:,0] = cv2.equalizeHist(Ecua_pot2[:,:,0])
Ecualizacion_pot2 = cv2.cvtColor(Ecua_pot2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_pot2)
ax[2, 1].set_title('Potencia Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Potencia con Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_pot2 = cv2.calcHist([Ecualizacion_pot2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_pot2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Potencia Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### CONJUNCION ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- CONJUNCION -------------------------------------|#

#//////CONJUNCION\\\\\\\\\\\\\\

opand = cv2.bitwise_and(res1,res2)

ax[0, 1].imshow(opand)
ax[0, 1].set_title('CONJUNCION')
ax[0, 1].axis('off')


#////////////Histograma de Conjuncion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_opand = cv2.calcHist([opand], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_opand, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Conjuncion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Conjuncion\\\\\\\\\\\\\\\\\\\\\\
Ecua_con = cv2.cvtColor(opand,cv2.COLOR_BGR2YUV)
Ecua_con[:,:,0] = cv2.equalizeHist(Ecua_con[:,:,0])
Ecualizacion_con = cv2.cvtColor(Ecua_con,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_con)
ax[2, 1].set_title('Conjuncion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Conjuncion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_con = cv2.calcHist([Ecualizacion_con], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_con, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma conjuncion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### DISYUNCION ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- DISYUNCION -------------------------------------|#

#//////DISYUNCION\\\\\\\\\\\\\\

opor = cv2.bitwise_or(res1,res2)

ax[0, 1].imshow(opor)
ax[0, 1].set_title('DISYUCION')
ax[0, 1].axis('off')


#////////////Histograma de Disyuncion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_dis = cv2.calcHist([opor], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_dis, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Disyucion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Disyucion\\\\\\\\\\\\\\\\\\\\\\
Ecua_dis = cv2.cvtColor(opor,cv2.COLOR_BGR2YUV)
Ecua_dis[:,:,0] = cv2.equalizeHist(Ecua_dis[:,:,0])
Ecualizacion_dis = cv2.cvtColor(Ecua_dis,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_dis)
ax[2, 1].set_title('Disyucion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Disyuncion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_dis = cv2.calcHist([Ecualizacion_dis], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_dis, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma Disyucion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### NEGACION ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- NEGACION -------------------------------------|#

#//////NEGACION\\\\\\\\\\\\\\

img1_neg = 1 - res1

ax[0, 1].imshow(img1_neg)
ax[0, 1].set_title('NEGACION')
ax[0, 1].axis('off')


#////////////Histograma de Negacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_neg = cv2.calcHist([img1_neg], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_neg, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Negacion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Negacion\\\\\\\\\\\\\\\\\\\\\\
Ecua_neg = cv2.cvtColor(img1_neg,cv2.COLOR_BGR2YUV)
Ecua_neg[:,:,0] = cv2.equalizeHist(Ecua_neg[:,:,0])
Ecualizacion_neg = cv2.cvtColor(Ecua_neg,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_neg)
ax[2, 1].set_title('Negacion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Negacion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_neg = cv2.calcHist([Ecualizacion_neg], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_neg, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Negacion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### NEGACION 2 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- NEGACION 2 -------------------------------------|#

#//////NEGACION\\\\\\\\\\\\\\

img2_neg = 1 - res2

ax[0, 1].imshow(img2_neg)
ax[0, 1].set_title('NEGACION 2')
ax[0, 1].axis('off')


#////////////Histograma de Negacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_neg2 = cv2.calcHist([img2_neg], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_neg2, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Negacion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Negacion\\\\\\\\\\\\\\\\\\\\\\
Ecua_neg2 = cv2.cvtColor(img2_neg,cv2.COLOR_BGR2YUV)
Ecua_neg2[:,:,0] = cv2.equalizeHist(Ecua_neg2[:,:,0])
Ecualizacion_neg2 = cv2.cvtColor(Ecua_neg2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_neg2)
ax[2, 1].set_title('Negacion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Negacion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_neg2 = cv2.calcHist([Ecualizacion_neg2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_neg2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Negacion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### TRASLACION ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRASLACION -------------------------------------|#

#//////TRASLACION\\\\\\\\\\\\\\

ancho = Img1.shape[1] #columnas
alto = Img1.shape[0] # filas
    
M = np.float32([[1,0,300],[0,1,250]]) #Construccion de la matriz
traslacion = cv2.warpAffine(Img1,M,(ancho,alto)) 

ax[0, 1].imshow(traslacion)
ax[0, 1].set_title('TRASLACION')
ax[0, 1].axis('off')


#////////////Histograma de Traslacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_tras = cv2.calcHist([traslacion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_tras, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Traslacion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Traslacion\\\\\\\\\\\\\\\\\\\\\\
Ecua_tras = cv2.cvtColor(traslacion,cv2.COLOR_BGR2YUV)
Ecua_tras[:,:,0] = cv2.equalizeHist(Ecua_tras[:,:,0])
Ecualizacion_tras = cv2.cvtColor(Ecua_tras,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_tras)
ax[2, 1].set_title('Traslacion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Traslacion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_tras = cv2.calcHist([Ecualizacion_tras], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_tras, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Traslacion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### TRASLACION 2 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRASLACION 2 -------------------------------------|#

#//////TRASLACION 2 \\\\\\\\\\\\\\

ancho = Img2.shape[1] #columnas
alto = Img2.shape[0] # filas
    
M = np.float32([[1,0,300],[0,1,250]]) #Construccion de la matriz
traslacion2 = cv2.warpAffine(Img2,M,(ancho,alto)) 

ax[0, 1].imshow(traslacion2)
ax[0, 1].set_title('TRASLACION 2')
ax[0, 1].axis('off')


#////////////Histograma de Traslacion 2 sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_tras = cv2.calcHist([traslacion2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_tras, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Traslacion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Traslacion\\\\\\\\\\\\\\\\\\\\\\
Ecua_tras2 = cv2.cvtColor(traslacion2,cv2.COLOR_BGR2YUV)
Ecua_tras2[:,:,0] = cv2.equalizeHist(Ecua_tras2[:,:,0])
Ecualizacion_tras2 = cv2.cvtColor(Ecua_tras2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_tras2)
ax[2, 1].set_title('Traslacion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Traslacion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_tras2 = cv2.calcHist([Ecualizacion_tras2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_tras2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Traslacion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### ESCALADO ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- ESCALADO -------------------------------------|#

#//////ESCALADO \\\\\\\\\\\\\\

Escalado = cv2.resize(res1, dsize=(480, 480)) 

ax[0, 1].imshow(Escalado)
ax[0, 1].set_title('Escalado')
ax[0, 1].axis('off')


#////////////Histograma de Escalado sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_esc = cv2.calcHist([Escalado], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_esc, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Escalado')
ax[1, 1].axis('off')

#///////////Ecualizacion de Escalado\\\\\\\\\\\\\\\\\\\\\\
Ecua_esc = cv2.cvtColor(Escalado,cv2.COLOR_BGR2YUV)
Ecua_esc[:,:,0] = cv2.equalizeHist(Ecua_esc[:,:,0])
Ecualizacion_esc = cv2.cvtColor(Ecua_esc,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_esc)
ax[2, 1].set_title('Escalado Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Escalado Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_esc = cv2.calcHist([Ecualizacion_esc], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_esc, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Escalado Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### ESCALADO 2 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- ESCALADO 2 -------------------------------------|#

#//////ESCALADO 2 \\\\\\\\\\\\\\

Escalado2 = cv2.resize(res2, dsize=(480, 480)) 

ax[0, 1].imshow(Escalado2)
ax[0, 1].set_title('Escalado')
ax[0, 1].axis('off')


#////////////Histograma de Escalado sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_esc2 = cv2.calcHist([Escalado2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_esc2, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Escalado')
ax[1, 1].axis('off')

#///////////Ecualizacion de Escalado\\\\\\\\\\\\\\\\\\\\\\
Ecua_esc2 = cv2.cvtColor(Escalado2,cv2.COLOR_BGR2YUV)
Ecua_esc2[:,:,0] = cv2.equalizeHist(Ecua_esc2[:,:,0])
Ecualizacion_esc2 = cv2.cvtColor(Ecua_esc2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_esc2)
ax[2, 1].set_title('Escalado Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Escalado Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_esc2 = cv2.calcHist([Ecualizacion_esc2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_esc2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Escalado Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### ROTACION ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- ROTACION -------------------------------------|#

#////// ROTACION \\\\\\\\\\\\\\

ancho = res1.shape[1] #columnas
alto = res1.shape[0] # filas
    
Rotacion = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
rota = cv2.warpAffine(res1,Rotacion,(ancho,alto))
 
ax[0, 1].imshow(rota)
ax[0, 1].set_title('ROTACION')
ax[0, 1].axis('off')


#////////////Histograma de Rotacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_rot = cv2.calcHist([rota], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_rot, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Rotacion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Rotacion\\\\\\\\\\\\\\\\\\\\\\
Ecua_rot = cv2.cvtColor(rota,cv2.COLOR_BGR2YUV)
Ecua_rot[:,:,0] = cv2.equalizeHist(Ecua_rot[:,:,0])
Ecualizacion_rot = cv2.cvtColor(Ecua_rot,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_rot)
ax[2, 1].set_title('Rotacion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Rotacion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_rot = cv2.calcHist([Ecualizacion_rot], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_rot, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Rotacion Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### ROTACION 2 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- ROTACION 2-------------------------------------|#

#////// ROTACION 2\\\\\\\\\\\\\\

ancho = res2.shape[1] #columnas
alto = res2.shape[0] # filas
    
Rotacion = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
rota2 = cv2.warpAffine(res2,Rotacion,(ancho,alto))
 
ax[0, 1].imshow(rota2)
ax[0, 1].set_title('ROTACION 2')
ax[0, 1].axis('off')


#////////////Histograma de Rotacion sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_rot2 = cv2.calcHist([rota2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_rot2, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Rotacion')
ax[1, 1].axis('off')

#///////////Ecualizacion de Rotacion\\\\\\\\\\\\\\\\\\\\\\
Ecua_rot2 = cv2.cvtColor(rota2,cv2.COLOR_BGR2YUV)
Ecua_rot2[:,:,0] = cv2.equalizeHist(Ecua_rot2[:,:,0])
Ecualizacion_rot2 = cv2.cvtColor(Ecua_rot2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_rot2)
ax[2, 1].set_title('Rotacion Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Rotacion Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_rot2 = cv2.calcHist([Ecualizacion_rot2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_rot2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Rotacion Ecualizada')
ax[3, 1].axis('off')
plt.show()
################################################################################
############################### TRASLACION AL FIN ##############################
################################# METODOLOGIA 1 ################################
##
#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRASLACION AL FIN -------------------------------------|#

#////// Traslacion al fin\\\\\\\\\\\\\\

ancho = res1.shape[1] #columnas
alto = res1.shape[0] # filas
M = np.float32([[1,0,100],[0,1,250]])
TF = cv2.warpAffine(res1,M,(ancho,alto))
 
ax[0, 1].imshow(TF)
ax[0, 1].set_title('TRASLACION AL FIN')
ax[0, 1].axis('off')


#////////////Histograma de Traslacion al Fin sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_tf = cv2.calcHist([TF], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_tf, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Traslacion al Fin')
ax[1, 1].axis('off')

#///////////Ecualizacion de Traslacion al Fin\\\\\\\\\\\\\\\\\\\\\\
Ecua_tf = cv2.cvtColor(TF,cv2.COLOR_BGR2YUV)
Ecua_tf[:,:,0] = cv2.equalizeHist(Ecua_tf[:,:,0])
Ecualizacion_tf = cv2.cvtColor(Ecua_tf,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_tf)
ax[2, 1].set_title('Traslacion al Fin Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Traslacion al Fin Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_tf = cv2.calcHist([Ecualizacion_tf], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_tf, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Traslacion al Fin Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################### TRASLACION AL FIN 2 ##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRASLACION AL FIN 2 -------------------------------------|#

#////// Traslacion al fin 2\\\\\\\\\\\\\\

ancho = res2.shape[1] #columnas
alto = res2.shape[0] # filas
M = np.float32([[1,0,100],[0,1,250]])
TF2 = cv2.warpAffine(res2,M,(ancho,alto))
 
ax[0, 1].imshow(TF2)
ax[0, 1].set_title('TRASLACION AL FIN 2')
ax[0, 1].axis('off')


#////////////Histograma de Traslacion al Fin sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_tf2 = cv2.calcHist([TF2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_tf2, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Traslacion al Fin')
ax[1, 1].axis('off')

#///////////Ecualizacion de Traslacion al Fin\\\\\\\\\\\\\\\\\\\\\\
Ecua_tf2 = cv2.cvtColor(TF2,cv2.COLOR_BGR2YUV)
Ecua_tf2[:,:,0] = cv2.equalizeHist(Ecua_tf2[:,:,0])
Ecualizacion_tf2 = cv2.cvtColor(Ecua_tf2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_tf2)
ax[2, 1].set_title('Traslacion al Fin Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Traslacion al Fin Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_tf2 = cv2.calcHist([Ecualizacion_tf2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_tf2, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Traslacion al Fin Ecualizada')
ax[3, 1].axis('off')
plt.show()

############################ METODOLOGIA 2 #################################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRASLACION AL FIN 3 -------------------------------------|#

#////// Traslacion al fin 3\\\\\\\\\\\\\\

rows,cols,ch = res1.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(res1,M,(300,300))
 
ax[0, 1].imshow(dst)
ax[0, 1].set_title('TRASLACION AL FIN 3')
ax[0, 1].axis('off')


#////////////Histograma de Traslacion al Fin sin Ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_tf3 = cv2.calcHist([dst], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_tf3, color = c)
    plt.xlim([0,256])
    
ax[1, 1].set_title('Histograma de Traslacion al Fin')
ax[1, 1].axis('off')

#///////////Ecualizacion de Traslacion al Fin\\\\\\\\\\\\\\\\\\\\\\
Ecua_tf3 = cv2.cvtColor(dst,cv2.COLOR_BGR2YUV)
Ecua_tf3[:,:,0] = cv2.equalizeHist(Ecua_tf3[:,:,0])
Ecualizacion_tf3 = cv2.cvtColor(Ecua_tf3,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_tf3)
ax[2, 1].set_title('Traslacion al Fin Ecualizada')
ax[2, 1].axis('off')
                                
#/////////////// Histograma de Traslacion al Fin Ecualizada \\\\\\\\\\\\\\\\\\\\  

for i, c in enumerate(color):
    hist2_ecua_tf3 = cv2.calcHist([Ecualizacion_tf3], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist2_ecua_tf3, color = c)
    plt.xlim([0,256])
    
ax[3, 1].set_title('Histograma de Traslacion al Fin Ecualizada')
ax[3, 1].axis('off')
plt.show()

#############################################################################
################################ TRASPUESTA #################################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRANSPUESTA -------------------------------------|#

#////// Traspuesta \\\\\\\\\\\\\\

Transpuesta1 = cv2.transpose(res1)

ax[0, 1].imshow(Transpuesta1)
ax[0, 1].set_title('Transpuesta 1')
ax[0, 1].axis('off')


#///////////Histograma Imagen TRANSPUESTA sin ecualizar \\\\\\\\\\\

for i, c in enumerate(color):
    hist_Transpuesta1 = cv2.calcHist([Transpuesta1], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Transpuesta 1')
ax[1, 1].axis('off')


#////////////////Ecualizacion imagen TRANSPUESTA\\\\\\\\\\\\\\\\\\\
Ecua_Transpuesta1 = cv2.cvtColor(Transpuesta1,cv2.COLOR_BGR2YUV)
Ecua_Transpuesta1[:,:,0] = cv2.equalizeHist(Ecua_Transpuesta1[:,:,0])
Ecualizacion_Transpuesta1 = cv2.cvtColor(Ecua_Transpuesta1,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Transpuesta1)
ax[2, 1].set_title('Imagen Transpuesta 1 Ecualizada')
ax[2, 1].axis('off')



#/////////////Histograma Imagen TRANSPUESTA ecualizada\\\\\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_ecua_Transpuesta1 = cv2.calcHist([Ecualizacion_Transpuesta1], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Transpuesta 1 Ecua')
ax[3, 1].axis('off')

plt.show()

################################ TRASPUESTA 2#################################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRANSPUESTA -------------------------------------|#

#////// Traspuesta \\\\\\\\\\\\\\

Transpuesta2 = cv2.transpose(res2)

ax[0, 1].imshow(Transpuesta2)
ax[0, 1].set_title('Transpuesta 2')
ax[0, 1].axis('off')


#///////////Histograma Imagen TRANSPUESTA sin ecualizar \\\\\\\\\\\

for i, c in enumerate(color):
    hist_Transpuesta2 = cv2.calcHist([Transpuesta2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Transpuesta2, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Transpuesta 2')
ax[1, 1].axis('off')


#////////////////Ecualizacion imagen TRANSPUESTA\\\\\\\\\\\\\\\\\\\
Ecua_Transpuesta2 = cv2.cvtColor(Transpuesta2,cv2.COLOR_BGR2YUV)
Ecua_Transpuesta2[:,:,0] = cv2.equalizeHist(Ecua_Transpuesta2[:,:,0])
Ecualizacion_Transpuesta2 = cv2.cvtColor(Ecua_Transpuesta2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Transpuesta2)
ax[2, 1].set_title('Imagen Transpuesta 2 Ecualizada')
ax[2, 1].axis('off')



#/////////////Histograma Imagen TRANSPUESTA ecualizada\\\\\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist_ecua_Transpuesta2 = cv2.calcHist([Ecualizacion_Transpuesta2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Transpuesta2, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Transpuesta 2 Ecua')
ax[3, 1].axis('off')

plt.show()

#############################################################################
################################ TRASPUESTA #################################
################################ METODOLOGIA 2##############################

#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- TRANSPUESTA -------------------------------------|#

#////// Traspuesta \\\\\\\\\\\\\\
def transponer(res1):
    t = []
    for i in range(len(res1[0])):
        t.append([])
        for j in range(len(res1)):
            t[i].append(res1[j][i])
    return t
Transpuesta1 = np.concatenate((res1,transponer(res1), res2), axis=1)

#Imagen TRANSPUESTA
ax[0, 1].imshow(Transpuesta1)
ax[0, 1].set_title('Transpuesta 3')
ax[0, 1].axis('off')


#Histograma Imagen TRANSPUESTA sin ecualizar

for i, c in enumerate(color):
    hist_Transpuesta1 = cv2.calcHist([Transpuesta1], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Transpuesta 3')
ax[1, 1].axis('off')


#Ecualizacion imagen TRANSPUESTA
Ecua_Transpuesta1 = cv2.cvtColor(Transpuesta1,cv2.COLOR_BGR2YUV)
Ecua_Transpuesta1[:,:,0] = cv2.equalizeHist(Ecua_Transpuesta1[:,:,0])
Ecualizacion_Transpuesta1 = cv2.cvtColor(Ecua_Transpuesta1,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Transpuesta1)
ax[2, 1].set_title('Imagen Transpuesta 3 Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRANSPUESTA ecualizada

for i, c in enumerate(color):
    hist_ecua_Transpuesta1 = cv2.calcHist([Ecualizacion_Transpuesta1], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Transpuesta 3 Ecua')
ax[3, 1].axis('off')

plt.show()

#############################################################################
################################ PROYECCION #################################


#|----------------------------- IMAGEN 1 ----------------------------------|#
Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

ax[0, 0].imshow(res1)
ax[0, 0].set_title('IMAGEN 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma de Imagen 1')
ax[1, 0].axis('off')

#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')

#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma de Imagen 1 Ecualizada')
ax[3, 0].axis('off')

#|------------------------------- IMAGEN 2 ---------------------------------|#

#//////Imagen 2\\\\\\\\\\\\\\

ax[0, 2].imshow(res2)
ax[0, 2].set_title('IMAGEN 2')
ax[0, 2].axis('off')


#////////////Histograma Imagen 2 sin ecualizar\\\\\\\\\\\\\

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
    
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')

#///////////Ecualizacion imagen 2\\\\\\\\\\\\\\\\\\\\\\
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')
                                
#Histograma Imagen 2 ecualizada  

for i, c in enumerate(color):
    hist2_ecua = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist2_ecua, color = c)
    plt.xlim([0,256])
    
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')

#|-------------------------------- PROYECCION -------------------------------------|#

#////// Proyeccion \\\\\\\\\\\\\\
proyeccion = dst + res1

#Imagen Proyeccion
ax[0, 1].imshow(proyeccion)
ax[0, 1].set_title('Proyeccion')
ax[0, 1].axis('off')


#Histograma Imagen Proyeccion sin ecualizar

for i, c in enumerate(color):
    hist_pro = cv2.calcHist([Proyeccion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_pro, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Proyeccion')
ax[1, 1].axis('off')


#Ecualizacion imagen Proyeccion
Ecua_pro = cv2.cvtColor(proyeccion,cv2.COLOR_BGR2YUV)
Ecua_pro[:,:,0] = cv2.equalizeHist(Ecua_pro[:,:,0])
Ecualizacion_pro = cv2.cvtColor(Ecua_pro,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_pro)
ax[2, 1].set_title('Imagen Proyeccion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen Proyeccion ecualizada

for i, c in enumerate(color):
    hist_ecua_pro = cv2.calcHist([Ecualizacion_pro], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_pro, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Proyeccion Ecua')
ax[3, 1].axis('off')

plt.show()

