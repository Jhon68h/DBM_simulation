// Corre el mismo rayo en CPU (OpenMP) y en GPU (CUDA) y compara tiempos.
// Uso: ./dbm [L] [eta] [semilla]   (por defecto 256 1 12345; hilos con OMP_NUM_THREADS)
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "dbm.hpp"

static void guardar_pgm(const Malla& m, const char* ruta) {
    std::ofstream f(ruta, std::ios::binary);
    f << "P5\n" << m.nx << ' ' << m.ny << "\n255\n";
    for (auto t : m.tipo) f.put(t == CANAL ? (char)255 : 0);
    if (!f) std::fprintf(stderr, "no se pudo escribir %s\n", ruta);
}

// Matriz ny×nx en texto, una fila por línea: se lee con numpy.loadtxt.
static void guardar_orden(const Resultado& r, int nx, const char* ruta) {
    std::ofstream f(ruta);
    for (std::size_t k = 0; k < r.orden.size(); ++k) f << r.orden[k] << ((k + 1) % nx ? ' ' : '\n');
    if (!f) std::fprintf(stderr, "no se pudo escribir %s\n", ruta);
}

int main(int argc, char** argv) {
    int L = argc > 1 ? std::atoi(argv[1]) : 256;
    float eta = argc > 2 ? std::atof(argv[2]) : 1.0f;
    unsigned semilla = argc > 3 ? std::atoi(argv[3]) : 12345;
    if (L < 4) {
        std::fprintf(stderr, "L debe ser >= 4\n");
        return 1;
    }
    std::printf("L = %d  eta = %g  semilla = %u  hilos OpenMP = %d\n\n", L, eta, semilla, omp_get_max_threads());

    Malla cpu = crear_malla(L), gpu = crear_malla(L), tmp = crear_malla(L);
    solve_cuda(tmp, 1.0f, 1.0f);  // calienta la GPU (crea el contexto) fuera de la medición
    Resultado rc = correr_rayo(cpu, solve_cpu, eta, semilla);
    Resultado rg = correr_rayo(gpu, solve_cuda, eta, semilla);

    std::printf("%-5s %7s %11s %12s %10s\n", "", "pasos", "iteraciones", "laplace (s)", "total (s)");
    std::printf("%-5s %7ld %11ld %12.3f %10.3f\n", "CPU", rc.pasos, rc.iteraciones, rc.t_laplace, rc.t_total);
    std::printf("%-5s %7ld %11ld %12.3f %10.3f\n", "CUDA", rg.pasos, rg.iteraciones, rg.t_laplace, rg.t_total);
    std::printf("\nspeedup CUDA/CPU: %.2fx (laplace)  %.2fx (total)\n", rc.t_laplace / rg.t_laplace, rc.t_total / rg.t_total);
    bool igual = cpu.tipo == gpu.tipo;

    std::filesystem::create_directories("salida");
    guardar_pgm(cpu, "salida/cpu.pgm");
    guardar_pgm(gpu, "salida/cuda.pgm");
    guardar_orden(rc, L, "salida/orden.txt");  
    return igual ? 0 : 1;
}
