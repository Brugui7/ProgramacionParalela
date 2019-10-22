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

/**
 * Shows all the options and calls the appropriate function depending of the chosen option
 */
void showMenu() {

    int option = 0;

    while (option != 3) {
        //Todo seleccionar algoritmo y mostrar menú distinto o más opciones por cada algoritmo
        printf("\n############### MENU SEMINARIO 0 ###############\n"
               "Indique qué acción desea realizar\n"
               "\t1. Suma de arrays\n"
               "\t2. Insertar manualmente\n"
               "\t3. Salir\n");
        printf("> ");

        scanf("%d", &option);
        fflush(stdin);
        switch (option) {
            case 1:
                sumArrays();
                break;
            case 3:
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
    int size = 0;
    int **array1, **array2, **arrayResult;
    printf("Introduzca el tamaño de los arrays\n> ");
    scanf("%d", &size);
    fflush(stdin);
    printf("Rellenando los arrays bidimensionales de %d X %d con valores aleatorios...\n", size, size);

    array1 = (int**) malloc(sizeof(int) * size);
    array2 = (int**) malloc(sizeof(int) * size);
    for (int i = 0; i < size; ++i) {
        array1[i] = (int*) malloc(sizeof(int) * size);
        array2[i] = (int*) malloc(sizeof(int) * size);
        for (int j = 0; j < size; ++j) {
            array1[i][j] = rand();
            array2[i][j] = rand();
        }
    }

    printf("Sumando los arrays...\n", size, size);

    arrayResult = (int**) malloc(sizeof(int) * size);
    for (int i = 0; i < size; ++i) {
        arrayResult[i] = (int*) malloc(sizeof(int) * size);
        for (int j = 0; j < size; ++j) {
            arrayResult[i][j] = array1[i][j] + array2[i][j];
        }
    }

    printf("Suma realizada correctamente, se muestran los resultados a continuación\n");

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("Resultado[%d][%d] = %d\n", i, j, arrayResult[i][j]);
        }
    }

    for (int i = 0; i < size; ++i) {
        free(array1[i]);
        free(array2[i]);
        free(arrayResult[i]);
    }
    free(array1);
    free(array2);
    free(arrayResult);
}

int main() {
    showMenu();
}