### Proyección de puntos en un video sobre el piso, dados puntos de referencia.
- Para la presente prueba de concepto omito la problemática de detectar a la persona, que se haría con algún modelo de visión, y simulo 1 solo frame del video sacando un screenshot y labeleando a mano la posición de los puntos de interés y de los gatos.
1. Coloco una webcam en altura elevada, simulando una cámara de vigilancia.
2. Coloco un cuadrado en el piso con cinta de papel, me servirá luego para determinar los puntos de referencia.
3. Armé un script para registrar el video de la webcam en tiempo real, y con la letra `c` se obtiene una captura de la imagen.
4. Usando `label-studio` genero los puntos de referencia y la posición de las patas delanteras de los gatos.
5. Levanto las imágenes y los labels, `points_ref` y `points_real` calculo la matriz de homografía usando `cv2.findHomography`.
6. Luego la utilizo para proyectar cualquier punto desde la imagen al plano real (por ejemplo, las patas del gato) usando `cv2.perspectiveTransform`.
7. Finalmente exporto los resultados en `/data/projected`.

<img src="measures.png" alt="measures.png" width="400"/><br>

<img src="data/projected/001.jpg" alt="001.jpg" width="400"/>
<img src="data/projected/002.jpg" alt="002.jpg" width="400"/>
<img src="data/projected/003.jpg" alt="003.jpg" width="400"/>
<img src="data/projected/004.jpg" alt="004.jpg" width="400"/>
<img src="data/projected/005.jpg" alt="005.jpg" width="400"/>
<img src="data/projected/006.jpg" alt="006.jpg" width="400"/>
<img src="data/projected/007.jpg" alt="007.jpg" width="400"/>
<img src="data/projected/008.jpg" alt="008.jpg" width="400"/>
<img src="data/projected/009.jpg" alt="009.jpg" width="400"/>
<img src="data/projected/010.jpg" alt="010.jpg" width="400"/>
<img src="data/projected/011.jpg" alt="011.jpg" width="400"/>

### label-studio
```bash
# Para arrancar el servidor y labelear en http://localhost:8080.
label-studio start
```

### Demo
- Stremear la webcam, si se da click en la imagen, guardar el frame en ese momento y en la posición donde se dió el click debe aparecer un punto rojo, y quizas un archivo csv para guardar nombre de la imagen, y posicion donde se dio el click. Para una demo.

### Ideas
- Detector de changuitos, si es que no están trackeados con algún dispositivo.


### Ver que pasa si:
- Los pies son tapados en la imagen.
- Solo se ve el torso.
- La persona salta.
