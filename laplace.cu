// Solver de Laplace en GPU: rojo-negro + SOR, el mismo método que solve_cpu (dbm.cpp).
//
// Solo esta parte del rayo corre en la GPU; candidatas y sorteo siguen en la CPU.
// En cada llamada: se sube φ y el tipo de celda, se itera hasta converger y se baja φ.
//
// Usa la misma fórmula que la CPU y se compila con --fmad=false (Makefile), así que da
// el mismo φ bit a bit y CPU y GPU hacen el mismo rayo.
#include <cstdio>
#include <cstdlib>

#include "dbm.hpp"

// Revisa el código de error de una llamada a CUDA. Si falló, dice qué, dónde, y termina.
#define CHECK(x)                                                                             \
    do {                                                                                     \
        cudaError_t e = (x);                                                                 \
        if (e != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA: %s en %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

const int REVISAR_CADA = 10;     // revisar el residuo cada 10 iteraciones (cada revisión hace esperar a la CPU)
const long MAX_ITER = 1000000;   // tope por si el solver nunca converge

// __device__: función que corre en la GPU y solo pueden llamar los kernels.
// Promedio de las 4 vecinas de (i, j). Bordes laterales Neumann: la vecina que no existe
// se reemplaza por la propia celda, así el campo no cruza el borde.
// Arriba y abajo siempre existen: los kernels no tocan la fila 0 (nube) ni la ny-1 (suelo).
__device__ inline float promedio(const float* p, int nx, int i, int j) {
    int il = i == 0 ? i : i - 1, ir = i == nx - 1 ? i : i + 1;
    return (p[j * nx + il] + p[j * nx + ir] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]) / 4.0f;
}

// __global__: kernel, lo lanza la CPU y lo ejecutan miles de hilos a la vez, uno por celda.
// Media iteración de rojo-negro: actualiza solo las celdas de aire del color indicado
// (0 = rojas, i + j par; 1 = negras, i + j impar). Una celda solo lee vecinas del otro
// color, así que todos los hilos pueden escribir a la vez sin pisarse.
__global__ void k_media_pasada(float* p, const std::uint8_t* t, int nx, int ny, int color, float omega) {
    // Posición de este hilo en la malla: su bloque × tamaño del bloque + su lugar en el bloque.
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    // Fuera de la malla (sobra en el último bloque), fila de nube o de suelo, u otro color: nada que hacer.
    if (i >= nx || j < 1 || j >= ny - 1 || (i + j) % 2 != color) return;
    int k = j * nx + i;
    if (t[k] != AIRE) return;  // canal, nube y suelo tienen φ fijo
    // SOR: φ + ω·(promedio − φ), escrito como (1 − ω)·φ + ω·promedio igual que en la CPU.
    // El recorte a [0, 1] evita que la sobre-relajación (ω ≈ 2) saque φ de rango.
    p[k] = fminf(fmaxf((1.0f - omega) * p[k] + omega * promedio(p, nx, i, j), 0.0f), 1.0f);
}

// Residuo: el mayor |promedio − φ| entre las celdas de aire. Si es chico, φ ya no cambia.
// Miles de hilos escriben en la misma variable `res`, así que se usa atomicMax, que compara
// y escribe sin que dos hilos se pisen. atomicMax solo acepta enteros: se guarda el float
// con sus bits leídos como int (__float_as_int). Para floats ≥ 0 esos enteros están en el
// mismo orden que los floats, y el residuo es un valor absoluto, así que siempre es ≥ 0.
__global__ void k_residuo(const float* p, const std::uint8_t* t, int nx, int ny, int* res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j < 1 || j >= ny - 1 || t[j * nx + i] != AIRE) return;
    atomicMax(res, __float_as_int(fabsf(promedio(p, nx, i, j) - p[j * nx + i])));
}

// Corre en la CPU y dirige a la GPU. Devuelve las iteraciones hechas; deja el φ resuelto en m.phi.
// shortcut: copia φ y tipo completos en cada paso del rayo (~0.3 MB con L = 256, poco frente
// al solver); si la copia llegara a pesar, dejar φ residente y subir solo la celda nueva.
long solve_cuda(Malla& m, float omega, float tol) {
    // Memoria de la GPU. `static` conserva los punteros entre llamadas: se reserva una sola vez
    // (o de nuevo si cambia el tamaño de la malla), no en cada paso del rayo.
    static float* p = nullptr;          // φ en la GPU
    static std::uint8_t* t = nullptr;   // tipo de celda en la GPU
    static int* res = nullptr;          // residuo (float guardado como int, ver k_residuo)
    static int n_reservado = 0;
    const int nx = m.nx, ny = m.ny, n = nx * ny;
    if (n != n_reservado) {
        cudaFree(p), cudaFree(t), cudaFree(res);  // cudaFree(nullptr) no hace nada
        CHECK(cudaMalloc(&p, n * sizeof(float)));
        CHECK(cudaMalloc(&t, n));
        CHECK(cudaMalloc(&res, sizeof(int)));
        n_reservado = n;
    }
    // CPU → GPU (host → device). Cada una tiene su propia memoria.
    CHECK(cudaMemcpy(p, m.phi.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t, m.tipo.data(), n, cudaMemcpyHostToDevice));

    // Bloques de 32×8 = 256 hilos. 32 columnas = un warp (32 hilos que la GPU ejecuta juntos):
    // leen celdas contiguas de una fila y la memoria se lee de una vez.
    // La rejilla redondea hacia arriba para cubrir la malla aunque L no sea múltiplo de 32 u 8.
    dim3 bloque(32, 8), rejilla((nx + 31) / 32, (ny + 7) / 8);
    for (long it = 1; it <= MAX_ITER; ++it) {
        // Los lanzamientos son asíncronos (la CPU no espera), pero la GPU los ejecuta en orden:
        // las negras empiezan cuando terminaron todas las rojas. Eso hace de barrera entre colores.
        k_media_pasada<<<rejilla, bloque>>>(p, t, nx, ny, 0, omega);  // rojas
        k_media_pasada<<<rejilla, bloque>>>(p, t, nx, ny, 1, omega);  // negras
        if (it % REVISAR_CADA) continue;

        CHECK(cudaMemset(res, 0, sizeof(int)));  // 0 como int es 0.0f como float
        k_residuo<<<rejilla, bloque>>>(p, t, nx, ny, res);
        float r;
        // Traer el residuo hace que la CPU espere a que la GPU termine todo lo pendiente; por eso
        // se revisa solo cada 10 iteraciones. También destapa errores de los kernels anteriores.
        CHECK(cudaMemcpy(&r, res, sizeof(float), cudaMemcpyDeviceToHost));
        if (r < tol) {
            CHECK(cudaMemcpy(m.phi.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));  // GPU → CPU
            return it;
        }
    }
    std::fprintf(stderr, "solve_cuda: no convergió\n");
    std::exit(1);
}
