#include <omp.h>
#include <sys/time.h>
#include <stdio.h>

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


int main() {
    int option = 0;

    while (option != 8) {

        printf("\n############### MENU TEMA 3 PARTE 1 ###############\n"
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
            case 8:
                printf("Saliendo...\n");
                break;
            default:
                printf("Por favor seleccione una opción válida\n");
                break;
        }
    }
}