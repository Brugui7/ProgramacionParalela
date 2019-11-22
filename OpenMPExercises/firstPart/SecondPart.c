

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
        exercise1Function1();

        #pragma omp section
        exercise1Function2();

    };

    gettimeofday(&end, NULL);
    timeInvested = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
    printf("-----------------\nTiempo invertido: %f\n-----------------\n", timeInvested);
}

void exercise2(){


}

void exercise3(){


}

void exercise4(){

}

void exercise5(){

}


void exercise6(){

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
                exercise6();
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