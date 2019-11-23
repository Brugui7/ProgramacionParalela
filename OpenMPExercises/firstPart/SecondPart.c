

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

void exercise1Function1(){
    int a = 0;
    int b = 3;
    for(int i = 0; i < 1000000000; i++){
        a += i*b;
    }
}

void exercise1Function2(){
    int a = 0;
    int b = 5;
    for(int i = 0; i < 1000000000; i++){
        a += i*b;
    }
}

void exercise1(){
    //For time measuring
    struct timeval start, end;
    double timeInvested;
    int threads = getThreadNumber();

    gettimeofday(&start, NULL);

    printf("Ejecutando funciones de manera lineal...\n");
    exercise1Function1();
    exercise1Function2();


    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("-----------------\nLineal Tiempo invertido: %f\n-----------------\n", timeInvested);

    printf("Ejecutando funciones de manera paralela...\n");
    omp_set_num_threads(threads);

    gettimeofday(&start, NULL);
    #pragma omp parallel sections
    {
        #pragma omp section
        { exercise1Function1(); }

        #pragma omp section
        {exercise1Function2(); }

    };

    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("-----------------\nTiempo invertido: %f\n-----------------\n", timeInvested);
}

void exercise2(){

    //For time measuring
    struct timeval start, end;
    double timeInvested;
    int size = 1073741824;

    double *x1, *x2, *x3;
    double y1 = 0.0f, y2 = 0.0f , y3 = 0.0f;
    x1 = (double*) malloc(sizeof(double) * size);
    x2 = (double*) malloc(sizeof(double) * size);
    x3 = (double*) malloc(sizeof(double) * size);

    int j = 0;

    printf("Rellenando arrays de tamaño %d...\n", size);

    for( int i = 0; i < size; i++){
        x1[i] = 1.0f;
        x2[i] = 1.0f;
        x3[i] = 1.0f;
    }

    printf("------------------------------------------\n"
           "Planificación estática"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);

        #pragma omp parallel for schedule(static) reduction(+:y1)
        for (j = 0; j < size; j++) {
            y1 += x1[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Planificación dinámica"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for schedule(dynamic) reduction(+:y2)
        for (j = 0; j < size; j++) {
            y2 += x2[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }

    printf("\n------------------------------------------\n"
           "Planificación guiada"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for schedule(guided)
        for (j = 0; j < size; j++) {
            y3 += x3[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f", timeInvested);
    }


    free(x1);
    free(x2);
    free(x3);
}

void exercise3(){

    //For time measuring
    struct timeval start, end;
    double timeInvested;

    double *x1, *x2, *x3;
    double y1 = 0.0f, y2 = 0.0f , y3 = 0.0f;
    x1 = (double*) malloc(sizeof(double) * 3000);
    x2 = (double*) malloc(sizeof(double) * 4000);
    x3 = (double*) malloc(sizeof(double) * 5000);

    int j = 0;

    printf("Rellenando arrays...\n");

    for(int i = 0; i < 3000; i++){
        x1[i] = 2.0f;
    }

    for(int i = 0; i < 4000; i++){
        x2[i] = 2.0f;
    }

    for(int i = 0; i < 5000; i++){
        x3[i] = 2.0f;
    }

    printf("\n\n\n******************************************\n"
           "USO DE REDUCTION"
           "\n******************************************\n");

    printf("------------------------------------------\n"
           "Array de tamaño 3000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        y1 = 0;
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);

        #pragma omp parallel for reduction(+:y1)
        for (j = 0; j < 3000; j++) {
            y1 += x1[j] * x1[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado: %f", timeInvested, y1);
    }

    printf("\n------------------------------------------\n"
           "Array de tamaño 4000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        y2 = 0;
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for reduction(+:y2)
        for (j = 0; j < 4000; j++) {
            y2 += x2[j] * x2[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado: %f", timeInvested, y2);
    }

    printf("\n------------------------------------------\n"
           "Array de tamaño 5000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        y3 = 0;
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel for reduction(+:y3)
        for (j = 0; j < 5000; j++) {
            y3 += x3[j] * x3[j];
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado: %f", timeInvested, y3);
    }


    printf("\n\n\n******************************************\n"
           "USO DE CRITICAL"
           "\n******************************************\n");

    printf("------------------------------------------\n"
           "Array de tamaño 3000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);

        #pragma omp parallel shared(x1) private(j)
        {
            #pragma omp critical (section1)
            {
                y1 = 0;
                for (j = 0; j < 3000; j++) {
                    y1 += x1[j] * x1[j];
                }
            }
        }


        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado: %f", timeInvested, y1);
    }

    printf("\n------------------------------------------\n"
           "Array de tamaño 4000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel shared(x2)
        {
            #pragma omp critical (section1)
            {
                y2 = 0;
                for (j = 0; j < 4000; j++) {
                    y2 += x2[j] * x2[j];
                }
            }

        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado: %f", timeInvested, y2);
    }

    printf("\n------------------------------------------\n"
           "Array de tamaño 5000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);
        #pragma omp parallel shared(x3)
        {
            #pragma omp critical (section1)
            {
                y3 = 0;
                for (j = 0; j < 5000; j++) {
                    y3 += x3[j] * x3[j];
                }
            }
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado: %f", timeInvested, y3);
    }


    free(x1);
    free(x2);
    free(x3);
}

void exercise4(){
    //For time measuring
    struct timeval start, end;
    double timeInvested;

    int threads = getThreadNumber();
    int size = 0, max, min;
    printf("Introduzca el tamaño del array\n> ");
    scanf("%d", &size);
    fflush(stdin);

    int *array = (int*) malloc(sizeof(int) * size);

    printf("Rellenando el array...\n");
    for (int i = 0; i < size; ++i) {
        array[i] = i;
    }
    max = min = array[0];

    gettimeofday(&start, NULL);

    #pragma omp parallel for shared(array) lastprivate(max, min) firstprivate(max, min)
    for (int i = 0; i < size; ++i) {
        if (array[i] > max) max = array[i];
        if (array[i] < min) min = array[i];

    }
    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("\tTiempo invertido: %f\t Máximo: %d\t Mínimo: %d", timeInvested, max, min);




}

void exercise5(){
    //For time measuring
    struct timeval start, end;
    double timeInvested;

    double **x1, **x2, **x3;
    double *y1, *y2, *y3;
    double *z1, *z2, *z3;
    x1 = (double**) malloc(sizeof(double) * 3000);
    x2 = (double**) malloc(sizeof(double) * 4000);
    x3 = (double**) malloc(sizeof(double) * 5000);
    y1 = (double*) malloc(sizeof(double) * 3000);
    y2 = (double*) malloc(sizeof(double) * 4000);
    y3 = (double*) malloc(sizeof(double) * 5000);
    z1 = (double*) malloc(sizeof(double) * 3000);
    z2 = (double*) malloc(sizeof(double) * 4000);
    z3 = (double*) malloc(sizeof(double) * 5000);

    int j = 0;

    printf("Rellenando arrays...\n");

    for(int i = 0; i < 3000; i++){
        x1[i] = (double*) malloc(sizeof(double) * 3000);
        y1[i] = 3.0f;
        z1[i] = 0;
        for (j = 0; j < 3000; ++j) {
            x1[i][j] = 2.0f;
        }
    }

    for(int i = 0; i < 4000; i++){
        x2[i] = (double*) malloc(sizeof(double) * 4000);
        y2[i] = 3.0f;
        z2[i] = 0;
        for (j = 0; j < 4000; ++j) {
            x2[i][j] = 2.0f;
        }
    }

    for(int i = 0; i < 5000; i++){
        x3[i] = (double*) malloc(sizeof(double) * 5000);
        y3[i] = 3.0f;
        z3[i] = 0;
        for (j = 0; j < 5000; ++j) {
            x3[i][j] = 2.0f;
        }
    }

    printf("------------------------------------------\n"
           "Array de tamaño 3000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);
        printf("\nNúmero de hilos: %d", i);

        omp_set_num_threads(i);
        #pragma omp parallel for collapse(2) shared(x1, y1)
        for (j = 0;j < 3000;j++) {
            for (int k = 0; k < 3000; ++k) {
                z1[j] += x1[j][k] * y1[k];
            }
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado en 2999: %f", timeInvested, z1[2999]);
    }

    printf("\n------------------------------------------\n"
           "Array de tamaño 4000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);
        printf("\nNúmero de hilos: %d", i);

        #pragma omp parallel for collapse(2) shared(x2, y2)
        for (j = 0;j < 4000;j++) {
            for (int k = 0; k < 4000; ++k) {
                z2[j] += x2[j][k] * y2[k];
            }
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado en 3999: %f", timeInvested, z2[3999]);
    }

    printf("\n------------------------------------------\n"
           "Array de tamaño 5000"
           "\n------------------------------------------");
    for (int i = 2; i < 10; i += 2) {
        gettimeofday(&start, NULL);

        omp_set_num_threads(i);

        printf("\nNúmero de hilos: %d", i);

        #pragma omp parallel for collapse(2) shared(x3, y3)
        for (j = 0;j < 5000;j++) {
            for (int k = 0; k < 5000; ++k) {
                z3[j] += x3[j][k] * y3[k];
            }
        }

        gettimeofday(&end, NULL);
        timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
        printf("\tTiempo invertido: %f\t Resultado en 4999: %f", timeInvested, z3[4999]);
    }



    for(int i = 0; i < 3000; i++){
        free(x1[i]);
    }

    for(int i = 0; i < 4000; i++){
        free(x2[i]);
    }

    for(int i = 0; i < 5000; i++){
        free(x3[i]);
    }

    free(x1);
    free(x2);
    free(x3);
    free(y1);
    free(y2);
    free(y3);
    free(z1);
    free(z2);
    free(z3);
}


void exercise6(){

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
                exercise6();
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