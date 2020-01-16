#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"

using namespace std;

/**
 * Solvation calculation kernel
 * @param atoms_r
 * @param atoms_l
 * @param nlig
 * @param rec_x_d
 * @param rec_y_d
 * @param rec_z_d
 * @param lig_x_d
 * @param lig_y_d
 * @param lig_z_d
 * @param ql_d
 * @param qr_d
 * @param energy_d
 * @param nconformations
 * @return
 */
__global__ void escalculation (int atoms_r, int atoms_l, int nlig, float *rec_x_d, float *rec_y_d, float *rec_z_d, float *lig_x_d, float *lig_y_d, float *lig_z_d, float *ql_d,float *qr_d, float *energy_d, int nconformations){
    int atomsLIdx = blockIdx.x * blockDim.x + threadIdx.x; // row
    int atomsRIdx = blockIdx.y * blockDim.y + threadIdx.y; // col

	float dist, total_elec = 0, miatomo[3], elecTerm;

    int totalAtomLig = nconformations * nlig;

    if (atomsLIdx < atoms_l && atomsRIdx < atoms_r){
        for (int i = 0; i < totalAtomLig; i += nlig) {
            miatomo[0] = *(lig_x_d + i + atomsLIdx);
            miatomo[1] = *(lig_y_d + i + atomsLIdx);
            miatomo[2] = *(lig_z_d + i + atomsLIdx);

            dist = calculaDistancia(rec_x_d[atomsRIdx], rec_y_d[atomsRIdx], rec_z_d[atomsRIdx], miatomo[0], miatomo[1], miatomo[2]);
            //__syncthreads();
            //energy_d[i / nlig] += (ql_d[atomsLIdx] * qr_d[atomsRIdx]) / dist;
            atomicAdd(&energy_d[i / nlig], (ql_d[atomsLIdx] * qr_d[atomsRIdx]) / dist);

        }

	}

}


/**
* Funcion para manejar el lanzamiento de CUDA
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	cudaError_t cudaStatus; //variable para recoger estados de cuda

	//seleccionamos device
	cudaSetDevice(0); //0 - Tesla K40 vs 1 - Tesla K230

	//creamos memoria para los vectores para GPU _d (device)
	float *rec_x_d, *rec_y_d, *rec_z_d, *qr_d, *lig_x_d, *lig_y_d, *lig_z_d, *ql_d, *energy_d;
	int memSizeRec      = sizeof(float) * atoms_r;
	int memSizeLig      = sizeof(float) * nlig * nconformations;
	int memSizeQl       = sizeof(float) * nlig;
    int memSizeEnergy   = sizeof(float) * nconformations;

    cudaStatus = cudaMalloc((void**)&energy_d, memSizeEnergy);
    if (cudaStatus != cudaSuccess){
        fprintf(stderr, "Error al reservar memoria en la GPU para energy\n");
        return;
    }

    // ############ REC ############
    cudaStatus = cudaMalloc((void**)&rec_x_d, memSizeRec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para rec_x_d\n");
        return;
    }

    cudaStatus = cudaMalloc((void**)&rec_y_d, memSizeRec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para rec_y_d\n");
        return;
    }

    cudaStatus = cudaMalloc((void**)&rec_z_d, memSizeRec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para rec_z_d\n");
        return;
    }

    cudaStatus = cudaMalloc((void**)&qr_d, memSizeRec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para qr_d\n");
        return;
    }

    // ############ LIG ############
    cudaStatus = cudaMalloc((void**)&ql_d, memSizeQl);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para ql\n");
        return;
    }

    cudaStatus = cudaMalloc((void**)&lig_x_d, memSizeLig);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para lig_x_d\n");
        return;
    }

    cudaStatus = cudaMalloc((void**)&lig_y_d, memSizeLig);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para lig_y_d\n");
        return;
    }

    cudaStatus = cudaMalloc((void**)&lig_z_d, memSizeLig);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU para lig_z_d\n");
        return;
    }

	//Pass data to the device
	cudaStatus = cudaMemcpy(energy_d, energy, memSizeEnergy, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de energy\n");
        return;
    }

    // ############ REC ############
    cudaStatus = cudaMemcpy(qr_d, qr, memSizeRec, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de qr\n");
        return;
    }

    cudaStatus = cudaMemcpy(rec_x_d, rec_x, memSizeRec, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de rec_x\n");
        return;
    }

    cudaStatus = cudaMemcpy(rec_y_d, rec_y, memSizeRec, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de rec_y\n");
        return;
    }

    cudaStatus = cudaMemcpy(rec_z_d, rec_z, memSizeRec, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de rec_z\n");
        return;
    }

    // ############ LIG ############
    cudaStatus = cudaMemcpy(ql_d, ql, memSizeQl, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de ql\n");
        return;
    }

    cudaStatus = cudaMemcpy(lig_x_d, lig_x, memSizeLig, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de lig_x\n");
        return;
    }

    cudaStatus = cudaMemcpy(lig_y_d, lig_y, memSizeLig, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de lig_y\n");
        return;
    }

    cudaStatus = cudaMemcpy(lig_z_d, lig_z, memSizeLig, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error al transferir información HtD de lig_z\n");
        return;
    }

	//Defines threads and blocks numbers
	int xBlockSize = 16;
    int yBlockSize = 16;
    int threadsPerBlock = xBlockSize * yBlockSize;

    dim3 block(ceil(atoms_l / xBlockSize), ceil(atoms_r / yBlockSize));
    dim3 thread(xBlockSize, yBlockSize);

	printf("Bloques x: %d\n", xBlockSize);
	printf("Bloques y: %d\n", yBlockSize);
	printf("Hilos por bloque: %d\n", threadsPerBlock);

	//llamamos a kernel
	escalculation <<< block,thread >>> (atoms_r, atoms_l, nlig, rec_x_d, rec_y_d, rec_z_d, lig_x_d, lig_y_d, lig_z_d, ql_d, qr_d, energy_d, nconformations);

	//control de errores kernel
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel %d\n", cudaStatus);

	//Gets the result back to the host
    cudaStatus = cudaMemcpy(energy, energy_d, memSizeEnergy, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess){
        fprintf(stderr, "Error al transferir información DtH de energy\n");
        return;
    }

	// para comprobar que la ultima conformacion tiene el mismo resultado que la primera
	printf("Termino electrostatico de conformacion %d es: %f\n", nconformations-1, energy[nconformations-1]);

	//resultado varia repecto a SECUENCIAL y CUDA en 0.000002 por falta de precision con float
	//posible solucion utilizar double, probablemente bajara el rendimiento -> mas tiempo para calculo
	printf("Termino electrostatico %f\n", energy[0]);

	//Liberamos memoria reservada para GPU
	cudaFree(rec_x_d);
	cudaFree(rec_y_d);
	cudaFree(rec_z_d);
	cudaFree(lig_x_d);
	cudaFree(lig_y_d);
	cudaFree(lig_z_d);
	cudaFree(ql_d);
    cudaFree(qr_d);
    cudaFree(energy_d);


}

/**
* Distancia euclidea compartida por funcion CUDA y CPU secuencial
*/
__device__ __host__ extern float calculaDistancia (float rx, float ry, float rz, float lx, float ly, float lz) {

  float difx = rx - lx;
  float dify = ry - ly;
  float difz = rz - lz;
  float mod2x=difx*difx;
  float mod2y=dify*dify;
  float mod2z=difz*difz;
  difx=mod2x+mod2y+mod2z;
  return sqrtf(difx);
}




/**
 * Funcion que implementa el termino electrostático en CPU
 */
void forces_CPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	double dist, total_elec = 0, miatomo[3], elecTerm;
    int totalAtomLig = nconformations * nlig;

	for (int k=0; k < totalAtomLig; k+=nlig){
	  for(int i=0;i<atoms_l;i++){
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);

			for(int j=0;j<atoms_r;j++){
                printf("i %d\tj %d\n", i, j);
                elecTerm = 0;
        dist=calculaDistancia (rec_x[j], rec_y[j], rec_z[j], miatomo[0], miatomo[1], miatomo[2]);
//				printf ("La distancia es %lf\n", dist);
        elecTerm = (ql[i]* qr[j]) / dist;
				total_elec += elecTerm;
//        printf ("La carga es %lf\n", total_elec);
			}
		}

		energy[k/nlig] = total_elec;
		total_elec = 0;
  }
	printf("Termino electrostatico %f\n", energy[0]);
}


extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql, float *qr, float *energy_desolv, int nconformaciones) {

	double elapsed_i, elapsed_o;

	switch (mode) {
		case 0://Sequential execution
			printf("\* CALCULO ELECTROSTATICO EN CPU *\n");
			printf("**************************************\n");
			printf("Conformations: %d\t Mode: %d, CPU\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_CPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("CPU Processing time: %f (seg)\n", elapsed_o);
			break;
		case 1: //OpenMP execution
			printf("\* CALCULO ELECTROSTATICO EN OPENMP *\n");
			printf("**************************************\n");
			printf("**************************************\n");
			printf("Conformations: %d\t Mode: %d, CMP\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("OpenMP Processing time: %f (seg)\n", elapsed_o);
			break;
		case 2: //CUDA exeuction
			printf("\* CALCULO ELECTROSTATICO EN CUDA *\n");
            printf("**************************************\n");
            printf("Conformaciones: %d\t Mode: %d, GPU\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_GPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("GPU Processing time: %f (seg)\n", elapsed_o);
			break;
	  	default:
 	    	printf("Wrong mode type: %d.  Use -h for help.\n", mode);
			exit (-1);
	}
}
