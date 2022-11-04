#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat obtenerImagen() { //Función para obtener la imagen de los archivos del proyecto, "Lenna.png" en este caso

	char NombreImagen[] = "Lenna.png"; //Cadena con la que identificar el archivo
	Mat imagen; //Matriz para almacenar imagen

	imagen = imread(NombreImagen, IMREAD_UNCHANGED); //Función de Opencv para leer un archivo de imagen ya sea especificando un nombre de archivo contenido en el proyecto
	// O especificando una ruta de acceso desde la computadora
	if (!imagen.data) //Condicional para comprobar que los datos de la imagen se hayan cargado correctamente o en su defecto notificar error
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1); //Salir del programa
	}

	return imagen; //retornamos la matriz con la imagen a operar
}

Mat gauss(int filasColumnas, float sigma) { //Función para crear e inicializar el kernel para filtro gaussiano

	int border = (filasColumnas - 1) / 2; //Variable de tipo entero para determinar los valores extremos x,y del kernel

	int variableY = border; //Variable de tipo entero para determinar los valores extremos y del kernel

	Mat matrizX(filasColumnas, filasColumnas, CV_32SC1);  //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a kernel solicitado, para valores de x
	Mat matrizY(filasColumnas, filasColumnas, CV_32SC1);  //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a kernel solicitado, para valores de y
	Mat matrizGaussiano(filasColumnas, filasColumnas, CV_32FC1);  //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a kernel solicitado, para valores de la formula

	for (int i = 0; i < filasColumnas; i++) { //filas
		int variableX = -1 * border; //Variable de tipo entero para determinar los valores extremos x del kernel
		for (int j = 0; j < filasColumnas; j++) { //columnas
			matrizX.at<int>(i, j) = variableX; //Inicializacion de los valores en x del kernel empezando por los extremos negativos a los positivos
			variableX++;
			matrizY.at<int>(i, j) = variableY; //Inicialización de los valores en y del kernel empezando por los extremos positivos a los negativos
		}
		variableY--;
	}

	for (int i = 0; i < filasColumnas; i++) { //filas
		for (int j = 0; j < filasColumnas; j++) { //columnas
			matrizGaussiano.at<float>(i, j) = (1 / (2 * CV_PI * pow(sigma, 2))) * exp(-1 * (pow(matrizX.at<int>(i, j), 2) + pow(matrizY.at<int>(i, j), 2)) / (2 * pow(sigma, 2)));
			//Inicialización de los valores del kernel empleando filtro gaussian
		}
	}

	float gaussianoPromedio = 0; //Variable para obtener la suma total del kernel
	for (int i = 0; i < filasColumnas; i++) {
		for (int j = 0; j < filasColumnas; j++) {
			gaussianoPromedio += matrizGaussiano.at<float>(i, j); //Suma de cada elemento del kernel gaussiano
		}
	}

	for (int i = 0; i < filasColumnas; i++) {
		for (int j = 0; j < filasColumnas; j++) {

			matrizGaussiano.at<float>(i, j) = matrizGaussiano.at<float>(i, j) / gaussianoPromedio; //Normalización del kernel gaussiano
			/*La normalización es simplemente una transformación lineal de su problema, normalmente necesaria debido a los límites impuestos por la aritmética de punto flotante.
			* Hay un número incontable de lugares donde puede ser necesaria.
			*/ 

		}
	}

	return matrizGaussiano; //retorno del kernel gaussiano para su proximo uso
}

Mat imagenBordes(Mat gris, int filasColumnas) {  //Funcion mediante la cual a la matriz original en escala de grises se le agregan bordes con 0 para evitar el desbordamiento del kernel gaussiano

	Mat destino(filasColumnas - 1 + gris.rows, filasColumnas - 1 + gris.cols, CV_8UC1, Scalar(0)); //Matriz de tipo sin signo a 8-bit de un solo canal con la adición del kernel gaussiano

	int border = (filasColumnas - 1) / 2; //Variable de tipo entero para copiar la imagen gris a la matriz con bordes inicializada en 0

	for (int i = 0; i < gris.rows; i++) {
		for (int j = 0; j < gris.cols; j++) {
			destino.at<uchar>(border + i, border + j) = gris.at<uchar>(i, j); //Inicialización de la matriz con bordes copiando la matriz original gris
		}
	}
	return destino; //Retorno de la matriz en escala de grises con los bordes necesarios
}

void suavizarImagen(Mat gauss, Mat matrizBordes, int filasColumnas, Mat& filtro) { //Función para inicializar la matriz suavizada con el filtro gaussiano mediante el kernel

	float Vtemp; //Variable para almacenar la suma de cada producto de submatriz por el kernel gaussiano

	Mat temp(filasColumnas, filasColumnas, CV_32FC1); //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a kernel solicitado, para obtener submatrices de imagen gris

	for (int i = 0; i < filtro.rows; i++) {
		for (int j = 0; j < filtro.cols; j++) {

			Vtemp = 0; //Inicialización de la variable en 0 para reinicar cada suma realizada
			temp = Mat(matrizBordes, Rect(j, i, filasColumnas, filasColumnas)); //Inicialización de la matriz temporal para crear cada submatriz mediante la función rect para cada indice

			for (int i = 0; i < filasColumnas; i++) {
				for (int j = 0; j < filasColumnas; j++) {

					Vtemp += temp.at<uchar>(i, j) * gauss.at<float>(i, j); //Suma de todos los valores del producto del kernel por la submatriz al momento

				}
			}

			filtro.at<uchar>(i, j) = Vtemp; //Inicialización de la matriz en cada indice, que contendra la imagen con el filtro aplicado o suavizada

		}
	}
}

void gradientes(Mat suavizada, Mat& Fx, Mat& Fy, Mat& G, Mat& Omega) { //Función para inicializar todas las matrices de gradiente a traves de la matriz Suavizada

	Mat hx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); //Matriz gradiente para valores de x
	Mat hy = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1); //Matriz gradiente para valores de y

	float x; //Matriz para la suma del producto de la matriz gradiente por submatrices de la matriz suavizada en x
	float y; //Matriz para la suma del producto de la matriz gradiente por submatrices de la matriz suavizada en y

	Mat temp(3, 3, CV_8UC1); //Matriz temporal tipo sin signo a 8-bit de un solo canal, tamaño igual a matrices gradiente, para cada submatriz de la matriz suavizada


	for (int i = 0; i < suavizada.rows - 2; i++) {
		for (int j = 0; j < suavizada.cols - 2; j++) {

			x = 0; //Inicialización de la variable en 0 para reinicar cada suma realizada
			y = 0; //Inicialización de la variable en 0 para reinicar cada suma realizada

			temp = Mat(suavizada, Rect(j, i, 3, 3)); //Inicialización de la matriz temporal para crear cada submatriz mediante la función rect para cada indice

			for (int i = 0; i < hx.rows; i++) {
				for (int j = 0; j < hx.cols; j++) {

					x += temp.at<uchar>(i, j) * hx.at<float>(i, j); //Suma de todos los valores del producto del kernel por la submatriz al momento
					y += temp.at<uchar>(i, j) * hy.at<float>(i, j); //Suma de todos los valores del producto del kernel por la submatriz al momento

				}
			}

			Fx.at<float>(i, j) = x; //Inicialización de la matriz con los valores del gradiente de x
			Fy.at<float>(i, j) = y; //Inicialización de la matriz con los valores del gradiente de y
			G.at<uchar>(i, j) = sqrt(pow(x,2) + pow(y, 2)); //Inicialización de la matriz con los valores del gradiente
			Omega.at<float>(i, j) = atan2(y, x); //Inicialización de la matriz con los valores del angulo del gradiente

		}
	}
}

void nonMaximumSuppresion(Mat& G, Mat& Omega, Mat& NMS) { //Funcion para aplicar nonMaximumSuppresion para la matriz gradiente mediante los angulos del gradiente	

	for (int i = 1; i < G.rows - 1; i++) {
		for (int j = 1; j < G.cols - 1; j++) {

			float angulo = abs(Omega.at<float>(i, j)); //Inicialización de variable angulo para cada indice de la matriz con los angulos del gradiente en valores absolutos
			uchar p = 255, q = 255; //Inicialización de variables de tipo uchar para inicializar los pixeles proximos al pixel procesado segun el angulo

			/* Nota: estas condiciones solo abarcan hasta 180° grados ya que con valores absolutos en los angulos y propiedades de los mismos
			* es posible comprobar el pixel en la dirección indicada por el angulo y el pixel opuesto a este sin pasar por angulos mayores a 180°
			*/

			if (0 <= angulo < 22.5 || 157.5 <= angulo <= 180) { //Condicional para determinar los pixeles situados a los costados del pixel a procesar en 0° y 180°
				p = G.at<uchar>(i, j + 1);
				q = G.at<uchar>(i, j - 1);
			}
			else if (22.5 <= angulo < 67.5) { //Condicional para determinar los pixeles situados a 45° del pixel a procesar
				p = G.at<uchar>(i + 1, j - 1);
				q = G.at<uchar>(i - 1, j + 1);
			}
			else if (67.5 <= angulo < 112.5) { //Condicional para determinar los pixeles situados a 90° del pixel a procesar
				p = G.at<uchar>(i + 1, j);
				q = G.at<uchar>(i - 1, j);
			}
			else if (112.5 <= angulo < 157.5) { //Condicional para determinar los pixeles situados a 135° del pixel a procesar
				p = G.at<uchar>(i - 1, j - 1);
				q = G.at<uchar>(i + 1, j + 1);
			}

			if (G.at<uchar>(i, j) >= q && G.at<uchar>(i, j) >= p) //Condicional para comprobar si los pixeles en esa dirección son más intensos que el procesado
				NMS.at<uchar>(i, j) = G.at<uchar>(i, j); //Inicialización de la matriz que contendra bordes en imagen más delgados por el proceso nonMaximumSuppresion
		}
	}
}

void Hysteresis(Mat& NMS, Mat& Canny) { //Función para inicializar matriz canny con los bordes detectados de la imagen original

	float bajoUmbralratio = 0.35; //Variable inicializada con el ratio sugerido por el material proporcionado para los umbrales bajos
	float altoUmbralratio = 0.9; //Variable inicializada con el ratio sugerido por el material proporcionado para los umbrales altos

	double minVal; //Variable para almacenar el minimo valor de una matriz
	double maxVal; //Variable para almacenar el máximo valor de una matriz
	Point minLoc; //Variable para almacenar la posición del minimo valor de una matriz
	Point maxLoc; //Variable para almacenar la posición del máximo valor de una matriz

	minMaxLoc(NMS, &minVal, &maxVal, &minLoc, &maxLoc); //Funcion para encontrar los valores minimos y maximos de una matriz asi como su ubicación.

	float altoUmbral = maxVal * altoUmbralratio; //Variable inicializada mediante el producto del valor más alto en la matriz con bordes delgados por el ratio sugerido
	float bajoUmbral = altoUmbral * bajoUmbralratio; //Variable inicializada mediante el producto del umbral más alto por el ratio sugerido

	int debil = bajoUmbral; //Variable inicializada con el umbral más bajo de la imagen para bordes de detalle
	int fuerte = 255; //Variable inicializada con el valor máximo para una imagen para bordes

	for (int i = 1; i < NMS.rows - 1; i++) {
		for (int j = 1; j < NMS.cols - 1; j++) {
			if (NMS.at<uchar>(i, j) >= altoUmbral) { //Condicional para determinar valores en la imagen con bordes delgados que superen el umbral más alto calculado
				Canny.at<uchar>(i, j) = fuerte; //Inicialización de la matriz de detección de bordes con un valor fuerte para bordes
			}
			else if (NMS.at<uchar>(i, j) < altoUmbral && NMS.at<uchar>(i, j) >= bajoUmbral) { //Condicional para determinar valores en la imagen con bordes delgados que se encuentren entre el umbral alto y el umbral bajo
				Canny.at<uchar>(i, j) = debil; //Inicialización de la matriz de detección de bordes con un valor debil para bordes de detalles
			}
		}
	}
}

void imprimir(Mat original, Mat gris_fun, Mat Suavizada, Mat matrizGaussiano, Mat G, Mat Fx, Mat Fy, Mat Omega, Mat NMS, Mat Canny) {
	//Función para imprimir todas las matrices empleadas, su tamaño y el kernel gaussiano empleado.

	namedWindow("Imagen Original", WINDOW_AUTOSIZE);
	imshow("Imagen Original", original); //Función para mostrar imagen original

	cout << "\nImagen Original Filas: " << original.rows << " Columnas: " << original.cols << endl; //Mostrar en consola el tamaño de la imagen original

	namedWindow("Imagen en grises", WINDOW_AUTOSIZE);
	imshow("Imagen en grises", gris_fun); //Función para mostrar imagen en grises

	cout << "\nImagen en Grises Filas: " << gris_fun.rows << " Columnas: " << gris_fun.cols << endl; //Mostrar en consola el tamaño de la imagen en grises

	namedWindow("Imagen suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen suavizada", Suavizada); //Función para mostrar imagen suavizada

	cout << "\nImagen Suavizada Filas: " << Suavizada.rows << " Columnas: " << Suavizada.cols << endl; //Mostrar en consola el tamaño de la imagen suavizada

	cout << endl;

	for (int i = 0; i < matrizGaussiano.rows; i++) { //Ciclos for para mostrar el kernel gaussiano empleado para la suavización de la imagen
		for (int j = 0; j < matrizGaussiano.cols; j++) {
			cout << matrizGaussiano.at<float>(i, j) << " ";
		}
		cout << endl;
	}

	namedWindow("Imagen aplicando Sobel", WINDOW_AUTOSIZE);
	imshow("Imagen aplicando Sobel", G); //Función para mostrar imagen aplicando sobel

	cout << "\nImagen aplicando Sobel Filas: " << G.rows << " Columnas: " << G.cols << endl; //Mostrar en consola el tamaño de la imagen aplicando sobel

	namedWindow("Imagen Fx", WINDOW_AUTOSIZE);
	imshow("Imagen Fx", Fx); //Función para mostrar imagen en gradiente de x

	cout << "\nImagen Fx Filas: " << Fx.rows << " Columnas: " << Fx.cols << endl; //Mostrar en consola el tamaño de la imagen en gradiente de x

	namedWindow("Imagen Fy", WINDOW_AUTOSIZE);
	imshow("Imagen Fy", Fy); //Función para mostrar imagen en gradiente de y

	cout << "\nImagen Fy Filas: " << Fy.rows << " Columnas: " << Fy.cols << endl; //Mostrar en consola el tamaño de la imagen en gradiente de y

	namedWindow("Imagen orientacion gradiente", WINDOW_AUTOSIZE);
	imshow("Imagen orientacion gradiente", Omega); //Función para mostrar imagen con orientación gradiente

	cout << "\nImagen orientacion gradiente Filas: " << Omega.rows << " Columnas: " << Omega.cols << endl; //Mostrar en consola el tamaño de la imagen con orientación gradiente

	namedWindow("Imagen NMS", WINDOW_AUTOSIZE);
	imshow("Imagen NMS", NMS); //Función para mostrar imagen aplicando NMS

	cout << "\nImagen NMS Filas: " << NMS.rows << " Columnas: " << NMS.cols << endl; //Mostrar en consola el tamaño de la imagen aplicando NMS

	namedWindow("Imagen deteccion de borde Canny", WINDOW_AUTOSIZE);
	imshow("Imagen deteccion de borde Canny", Canny); //Función para mostrar imagen con detección de borde Canny

	cout << "\nImagen deteccion de borde Canny Filas: " << Canny.rows << " Columnas: " << Canny.cols << endl; //Mostrar en consola el tamaño de la imagen con detección de borde Canny
}

int main() {

	Mat imagen = obtenerImagen(); //Función para obtener la imagen de los archivos del proyecto, "Lenna.png" en este caso

	int rows = imagen.rows; //Variable tipo entera del numero de filas de la imagen para futuras operaciones 512
	int columns = imagen.cols; //Variable tipo entera del numero de columnas de la imagen para futuras operaciones 512

	int filasColumnas = 0; //Varibale tipo entera para determinar las filas y columnas de un kernel dinamico gaussiano
	float sigma = 0; //Variable tipo flotante para la operación de los valores del kernel gaussiano

	cout << "Ingrese el numero de filas y columnas: "; cin >> filasColumnas; //Entrada de la variable filas columnas
	cout << "Ingrese el sigma: "; cin >> sigma; //Entrada de la variable sigma

	Mat gris_fun(rows, columns, CV_8UC1); //Matriz de tipo sin signo a 8-bit de un solo canal, tamaño igual a imagen original

	cvtColor(imagen, gris_fun, COLOR_BGR2GRAY); //función de opencv para convertir una imagen de 3 canales a uno de un canal en escala de grises

	Mat matrizGaussiano = gauss(filasColumnas, sigma); //matriz que contiene el kernel gaussiano con su funcion para crearla

	Mat matrizBordes = imagenBordes(gris_fun, filasColumnas); //matriz en escala de grises adecuada con bordes para la aplicación del filtro gaussiano

	Mat Suavizada(rows, columns, CV_8UC1, Scalar(0)); //Matriz de tipo sin signo a 8-bit de un solo canal, tamaño igual a imagen original, para almacenar matriz con filtro gaussiano aplicado o suavizada.

	suavizarImagen(matrizGaussiano, matrizBordes, filasColumnas, Suavizada); //Función para inicializar la matriz suavizada con el filtro gaussiano mediante el kernel

	Mat Fx(rows, columns, CV_32FC1, Scalar(0)); //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a imagen original, para valores del gradiente x
	Mat Fy(rows, columns, CV_32FC1, Scalar(0)); //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a imagen original, para valores del gradiente y
	Mat G(rows, columns, CV_8UC1, Scalar(0)); //Matriz de tipo sin signo a 8-bit de un solo canal, tamaño igual a imagen original, para valores del gradiente
	Mat Omega(rows, columns, CV_32FC1, Scalar(0)); //Matriz de Profundidad con signo a 32 bits en un canal, tamaño igual a imagen original, para angulo del gradiente

	gradientes(Suavizada, Fx, Fy, G, Omega); //Función para inicializar todas las matrices de gradiente a traves de la matriz Suavizada

	Mat NMS(G.rows, G.cols, CV_8UC1, Scalar(0)); //Matriz de tipo sin signo a 8-bit de un solo canal, tamaño igual a imagen original, para aplicación del metodo nonMaximumSuppresion y adelgazar los bordes de la matriz gradiente

	nonMaximumSuppresion(G, Omega, NMS); //Funcion para aplicar nonMaximumSuppresion para la matriz gradiente mediante los angulos del gradiente

	Mat Canny(NMS.rows, NMS.cols, CV_8UC1, Scalar(0)); //Matriz de tipo sin signo a 8-bit de un solo canal, tamaño igual a imagen original, para aplicación de umbrales e histéresis para detectar bordes en imagen

	Hysteresis(NMS, Canny); //Función para inicializar matriz canny con los bordes detectados de la imagen original

	imprimir(imagen,gris_fun,Suavizada,matrizGaussiano,G,Fx,Fy,Omega,NMS,Canny); //Función para imprimir todas las matrices empleadas, su tamaño y el kernel gaussiano empleado.

	waitKey(0); //Función para hacer esperar al programa hasta teclear una tecla

	return 1; //retorno a 1 de la función main
}