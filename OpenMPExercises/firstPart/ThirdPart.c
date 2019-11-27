////////////////////////////
////////////////////////////
////////////////////////////
/**
 * Last part of the exercises
 * @author Alejandro Brugarolas
 * @since 2019-11
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>


void jacobiSolver();
void doJacobi(float *array1, float *array2, int arraySize);
void doParallelJacobi1(float *array1, float *array2, int arraySize);
void doParallelJacobi2(float *array1, float *array2, int arraySize);


/**
 * Asks for the number of threads to be used in the function
 * @return int
 */
int getThreadNumber() {
    int threadNumber;
    printf("Introduzca el número de hilos a ejecutar\n >");
    scanf("%d", &threadNumber);
    fflush(stdin);
    return threadNumber;
}

/**
* Asks the user for a size and a number or iterations then sums two "2D" (using row major order) arrays of sizeXsize ints
*/
void jacobiSolver(){
    int threads = getThreadNumber();
    //For time measuring
    struct timeval start, end;
    double timeInvested;

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

    printf("Aplicando el método de manera secuencial...\n");
    omp_set_num_threads(threads);

    gettimeofday(&start, NULL);
    //The Jacobi's method itself
    float *aux;
    for (int k = 0; k < iterations; ++k) {
        doJacobi(array1, array2, size);
        aux = array1;
        array1 = array2;
        array2 = aux;
    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;

    printf("Método aplicado correctamente\tTiempo invertido: %f.\n\n", timeInvested);


    printf("Aplicando el método de manera paralela con un for...\n");

    gettimeofday(&start, NULL);
    //The Jacobi's method itself
    for (int k = 0; k < iterations; ++k) {
        doParallelJacobi1(array1, array2, size);
        aux = array1;
        array1 = array2;
        array2 = aux;
    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;

    printf("Método aplicado correctamente\tTiempo invertido: %f.\n\n", timeInvested);

    printf("Aplicando el método de manera paralela con un for nested...\n");

    gettimeofday(&start, NULL);
    //The Jacobi's method itself
    for (int k = 0; k < iterations; ++k) {
        doParallelJacobi2(array1, array2, size);
        aux = array1;
        array1 = array2;
        array2 = aux;
    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;

    printf("Método aplicado correctamente\tTiempo invertido: %f.\n", timeInvested);
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
 * Applies the Jacobi's method itselfs
 * @param array1
 * @param array2
 * @param arraySize the size of the arrays
 */
void doParallelJacobi1(float *array1, float *array2, int arraySize){
    int position = 0; //Aux variable to calculate a position in the 2D array
    #pragma omp parallel for schedule(static) shared(arraySize) firstprivate(array2, position)
    for (int i = 1; i < arraySize - 1; ++i) {
        for (int j = 1; j < arraySize - 1; ++j) {
            position = i * arraySize + j;
            array2[position] = 0.2f * (array1[position] + array1[position - 1] + array1[position + 1] + array1[position + arraySize] + array1[position - arraySize]);
        }
    }
}

/**
 * Applies the Jacobi's method itselfs
 * @param array1
 * @param array2
 * @param arraySize the size of the arrays
 */
void doParallelJacobi2(float *array1, float *array2, int arraySize){
    int position = 0; //Aux variable to calculate a position in the 2D array
    #pragma omp parallel for collapse(2) shared(array1) firstprivate(array2, position)
    for (int i = 1; i < arraySize - 1; ++i) {
        for (int j = 1; j < arraySize - 1; ++j) {
            position = i * arraySize + j;
            array2[position] = 0.2f * (array1[position] + array1[position - 1] + array1[position + 1] + array1[position + arraySize] + array1[position - arraySize]);
        }
    }
}

int main() {
    int option = 0;

    while (option != 7) {

        printf("\n\n\n\n############### MENU TEMA 3 PARTE 1 ###############\n"
               "Indique qué acción desea realizar\n"
               "\t1. Ejercicio 1\n"
               "\t2. Ejercicio 2\n"
               "\t3. Ejercicio 3\n"
               "\t4. Ejercicio 4\n"
               "\t5. Ejercicio 5\n"
               "\t6. Ejercicio 6\n"
               "\t7. Salir\n");
        printf("> ");

        scanf("%d", &option);
        fflush(stdin);
        switch (option) {
            case 1:
                jacobiSolver();
                break;
            case 2:
                break;
            case 3:
                break;
            case 4:
                break;
            case 5:
                break;
            case 6:
                break;
            case 7:
                printf("Saliendo...\n");
                break;
            default:
                printf("Por favor seleccione una opción válida\n");
                break;
        }
    }
}