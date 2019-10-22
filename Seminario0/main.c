/**
 * @file
 * @brief Task 0
 * @author Alejandro Brugarolas
 * @since 2019-10
 */

#include <stdio.h>
#include <stdlib.h>

void showMenu();
void sumArrays();
void jacobiSolver();
void doJacobi(float *array1, float *array2, int arraySize);
void sumFloatArray();

/**
 * Shows all the options and calls the appropriate function depending of the chosen option
 */
void showMenu() {

    int option = 0;

    while (option != 4) {
        //Todo seleccionar algoritmo y mostrar menú distinto o más opciones por cada algoritmo
        printf("\n############### MENU SEMINARIO 0 ###############\n"
               "Indique qué acción desea realizar\n"
               "\t1. Suma de arrays\n"
               "\t2. Método de Jacobi\n"
               "\t3. Suma Array de Float\n"
               "\t4. Salir\n");
        printf("> ");

        scanf("%d", &option);
        fflush(stdin);
        switch (option) {
            case 1:
                sumArrays();
                break;
            case 2:
                jacobiSolver();
                break;
            case 3:
                sumFloatArray();
                break;
            case 4:
                printf("Saliendo...");
                break;
            default:
                printf("Por favor seleccione una opción válida\n");
                break;
        }
    }
}

/**
 * Asks the user for a size then sums two 2D arrays of sizeXsize ints
 * This can be optimized by removing some loops
 */
void sumArrays(){
    int size = 0, showResult = 0;
    int position = 0; //Aux variable to calculate a position in the 2D array
    int *array1, *array2, *arrayResult;
    printf("¿Desea mostrar el resultado de la suma por pantalla?\n\t1. Sí\n\t2. No\n> ");
    scanf("%d", &showResult);
    fflush(stdin);

    printf("Introduzca el tamaño de los arrays\n> ");
    scanf("%d", &size);
    fflush(stdin);
    printf("Rellenando los arrays bidimensionales de %d X %d con valores aleatorios...\n", size, size);

    //Allocates the memory
    array1 = (int*) malloc(sizeof(int) * size * size);
    array2 = (int*) malloc(sizeof(int) * size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            position = i * size + j;
            array1[position] = rand();
            array2[position] = rand();
        }
    }

    printf("Sumando los arrays...\n", size, size);

    arrayResult = (int*) malloc(sizeof(int) * size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            position = i * size + j;
            arrayResult[position] = array1[position] + array2[position];
        }
    }

    if(showResult == 1){
        printf("Suma realizada correctamente, se muestran los resultados a continuación\n");
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                position = i * size + j;
                printf("Resultado[%d][%d] = %d\n", i, j, arrayResult[position]);
            }
        }
    }

    //Releases the memory previously allocated
    free(array1);
    free(array2);
    free(arrayResult);
}

/**
 * Asks the user for a size and a number or iterations then sums two "2D" (using row major order) arrays of sizeXsize ints
 */
void jacobiSolver(){
    int size = 0, iterations = 0;
    int position = 0; //Aux variable to calculate a position in the 2D array
    float *array1, *array2;

    printf("Introduzca el número de iteraciones\n> ");
    scanf("%d", &iterations);
    fflush(stdin);

    printf("Introduzca el tamaño de los arrays\n> ");
    scanf("%d", &size);
    fflush(stdin);

    //Allocates the memory
    printf("Rellenando los arrays de %d X %d...\n", size, size);
    array1 = (float*) malloc(sizeof(float) * size * size);
    array2 = (float*) malloc(sizeof(float) * size * size);

    //Fills the arrays with 150 in the first and last position of each row and 70 in the rest
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            position = i * size + j;
            array1[position] = array2[position] = i == 0  || j == 0 || i == size - 1 || j == size - 1 ? 150.0f : 70.0f;
        }
    }

    printf("Aplicando el método...\n");
    //The Jacobi's method itself
    float *aux;
    for (int k = 0; k < iterations; ++k) {
        doJacobi(array1, array2, size);
        aux = array1;
        array1 = array2;
        array2 = aux;
    }

    printf("Método aplicado correctamente.\n");

    //Releases the memory previously allocated
    free(array1);
    free(array2);
}

/**
 * Applies the Jacobi's method itselfs
 * @param array1
 * @param array2
 * @param arraySize the size of the arrays
 */
void doJacobi(float *array1, float *array2, int arraySize){
    int position = 0; //Aux variable to calculate a position in the 2D array
    for (int i = 1; i < arraySize - 1; ++i) {
        for (int j = 1; j < arraySize - 1; ++j) {
            position = i * arraySize + j;
            array2[position] = 0.2f * (array1[position] + array1[position - 1] + array1[position + 1] + array1[position + arraySize] + array1[position - arraySize]);
        }
    }
}



/**
 * Asks the user for a size then sums two 1D array of random floats
 */
void sumFloatArray(){
    int size = 0;
    float result = 0.0f;
    float *array;

    printf("Introduzca el tamaño del array\n> ");
    scanf("%d", &size);
    fflush(stdin);
    printf("Rellenando el array...\n");

    //Allocates the memory
    array = (float *) malloc(sizeof(float) * size);

    for (int i = 0; i < size; ++i) {
        array[i] = 1.0f;
    }

    printf("Sumando los %d valores del array...\n", size);


    for (int i = 0; i < size; ++i) {
        result += array[i];
    }

    printf("Suma realizada correctamente, el resultado es: %f\n", result);

    //Releases the memory previously allocated
    free(array);
}

int main() {
    showMenu();
}