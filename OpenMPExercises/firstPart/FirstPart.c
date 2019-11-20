#include <omp.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Asks for the number of threads to be used in the function
 * @return int
 */
int getThreadNumber(){
    int threadNumber;
    printf("Introduzca el número de hilos a ejecutar\n >");
    scanf("%d", &threadNumber);
    fflush(stdin);
    return threadNumber;
}

void exercise1(){
    omp_set_num_threads(getThreadNumber());
    int id;
    #pragma omp parallel private(id)
    {
        id = omp_get_thread_num();
        printf("Soy el hilo %d.\n", id);

        #pragma omp master
        {
            printf("Se están ejecutando %d hilos en paralelo\n", omp_get_num_threads());
        }
    }
}

void exercise2(){
    omp_set_num_threads(getThreadNumber());
    int number = 10;
    printf("El valor de la variable es %d antes de empezar el framento paralelizado\n---------------------------\n", number);
    #pragma omp parallel for firstprivate(number)
        for (int i = 0; i < 3; i++){
            printf("Soy un hilo, y con la directiva firstPrivate, puedo seguir viendo el valor inicial de la variable: %d\n"
                   "Ahora cambiaré el valor de la variable\n", number++);
        }

    printf("----------------------------------\n Al final de la parte paralelizada, no se ven las modificaciones hechas a la variable, por lo que el valor continúa como %d\n", number);

    #pragma omp parallel for lastprivate(number)
        for (int i = 0; i < 3; i++){
            printf("Soy un hilo, y con la directiva lastPrivate, no puedo seguir viendo el valor inicial de la variable: %d\n"
                   "Ahora cambiaré el valor de la variable\n", number);
            number = 0 + i;
        }

    printf("----------------------------------\n Al final de la parte paralelizada, se ven las modificaciones hechas a la variable, por lo que el valor ahora es %d\n", number);

}

void exercise3(){
    //For time measuring
    struct timeval start, end;
    double timeInvested;

    int threadNumber = getThreadNumber();
    int firstArray[1024], secondArray[2048], lastArray[4096];
    int firstResult = 0, secondResult = 0, lastResult = 0;
    printf("Rellenando los arrays...\n");
    for (int i = 0; i < 1024; i++){
        firstArray[i] = i;
    }
    for (int i = 0; i < 2048; i++){
        secondArray[i] = i;
    }
    for (int i = 0; i < 4096; i++){
        lastArray[i] = i;
    }

    gettimeofday(&start, NULL);
    omp_set_num_threads(threadNumber);
    #pragma omp parallel for reduction(+:firstResult)
    for (int i = 0; i < 1024; i++){
        firstResult += firstArray[i];
    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("Terminada la suma del vector de 1024 elementos, resultado: %d\n", firstResult);
    printf("Tiempo invertido: %f\n", timeInvested);

    gettimeofday(&start, NULL);
    #pragma omp parallel for reduction(+:secondResult)
    for (int i = 0; i < 2048; i++) {
        secondResult += secondArray[i];
    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("Terminada la suma del vector de 2048 elementos, resultado: %d\n", secondResult);
    printf("Tiempo invertido: %f\n", timeInvested);

    gettimeofday(&start, NULL);
    #pragma omp parallel for reduction(+:lastResult)
    for (int i = 0; i < 4096; i++) {
        lastResult += lastArray[i];
    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("Terminada la suma del vector de 4096 elementos, resultado: %d\n", lastResult);
    printf("Tiempo invertido: %f\n", timeInvested);


}

void exercise4(){
    //For time measuring
    struct timeval start, end;
    double timeInvested;

    double *x1, *x2, *x3, *x4, *y1, *y2, *y3, *y4;
    double alpha = 3.14f;
    x1 = (double*) malloc(sizeof(double) * 1024);
    y1 = (double*) malloc(sizeof(double) * 1024);
    x2 = (double*) malloc(sizeof(double) * 2048);
    y2 = (double*) malloc(sizeof(double) * 2048);
    x3 = (double*) malloc(sizeof(double) * 4096);
    y3 = (double*) malloc(sizeof(double) * 4096);
    x4 = (double*) malloc(sizeof(double) * 1048576);
    y4 = (double*) malloc(sizeof(double) * 1048576);

    int j = 0;

    printf("Rellenando arrays...\n");

    for( int i = 0; i < 1024; i++){
        x1[i] = y1[i] = 1.0f;
    }

    for( int i = 0; i < 2048; i++){
        x2[i] = y2[i] = 1.0f;
    }

    for( int i = 0; i < 4096; i++){
        x3[i] = y3[i] = 1.0f;
    }

    for( int i = 0; i < 1048576; i++){
        x4[i] = y4[i] = 1.0f;
    }

    printf("------------------------------------------\n"
           "Array de 1024 elementos"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for private(j) firstprivate(alpha, x1, y1) lastprivate(y1)
            for (j = 0; j < 1024; j++) {
                y1[j] = x1[j] * alpha + y1[j];
            }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Array de 2048 elementos"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for private(j) shared(y2) firstprivate(alpha, x2)
        for (j = 0; j < 2048; j++) {
            y2[j] = x2[j] * alpha + y2[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Array de 4096 elementos"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for private(j) shared(y3) firstprivate(alpha, x3)
        for (j = 0; j < 4096; j++) {
            y3[i] = x3[j] * alpha + y3[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Array de 1048576 elementos\n"
           "------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for private(j) shared(y4) firstprivate(alpha, x4)
        for (j = 0; j < 1048576; j++) {
            y4[j] = x4[j] * alpha + y4[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }


    free(x1);
    free(x2);
    free(x3);
    free(x4);
    free(y1);
    free(y2);
    free(y3);
    free(y4);
}

void exercise5(){
    //For time measuring
    struct timeval start, end;
    double timeInvested;

    double *x1, *x2, *x3, *x4;
    double y1 = 0.0f, y2 = 0.0f , y3 = 0.0f, y4 = 0.0f;
    x1 = (double*) malloc(sizeof(double) * 1024);
    x2 = (double*) malloc(sizeof(double) * 2048);
    x3 = (double*) malloc(sizeof(double) * 4096);
    x4 = (double*) malloc(sizeof(double) * 1048576);

    int j = 0;

    printf("Rellenando arrays...\n");

    for( int i = 0; i < 1024; i++){
        x1[i] = 1.0f;
    }

    for( int i = 0; i < 2048; i++){
        x2[i] = 1.0f;
    }

    for( int i = 0; i < 4096; i++){
        x3[i] = 1.0f;
    }

    for( int i = 0; i < 1048576; i++){
        x4[i] = 1.0f;
    }

    printf("------------------------------------------\n"
           "Array de 1024 elementos"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for reduction(+:y1)
            for (j = 0; j < 1024; j++) {
                y1 += x1[j];
            }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Array de 2048 elementos"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for reduction(+:y2)
        for (j = 0; j < 2048; j++) {
            y2 += x2[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Array de 4096 elementos"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for reduction(+:y3)
        for (j = 0; j < 4096; j++) {
            y3 += x3[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Array de 1048576 elementos\n"
           "------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for reduction(+:y4)
        for (j = 0; j < 1048576; j++) {
            y4 += x4[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }


    free(x1);
    free(x2);
    free(x3);
    free(x4);
}


int main() {
    int option = 0;

    while (option != 8) {

        printf("\n\n\n\n############### MENU TEMA 3 PARTE 1 ###############\n"
               "Indique qué acción desea realizar\n"
               "\t1. Ejercicio 1\n"
               "\t2. Ejercicio 2\n"
               "\t3. Ejercicio 3\n"
               "\t4. Ejercicio 4\n"
               "\t5. Ejercicio 5\n"
               "\t6. Ejercicio 6\n"
               "\t7. Ejercicio 7\n"
               "\t8. Salir\n");
        printf("> ");

        scanf("%d", &option);
        fflush(stdin);
        switch (option) {
            case 1:
                exercise1();
                break;
            case 2:
                exercise2();
                break;
            case 3:
                exercise3();
                break;
            case 4:
                exercise4();
                break;
            case 5:
                exercise5();
                break;
            case 6:
                //exercise3();
                break;
            case 8:
                printf("Saliendo...\n");
                break;
            default:
                printf("Por favor seleccione una opción válida\n");
                break;
        }
    }
}