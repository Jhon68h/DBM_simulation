// DBM 2D: nube arriba (φ = 0), suelo abajo (φ = 1), canal en φ = 0.
#pragma once

#include <cstdint>
#include <vector>

enum : std::uint8_t { AIRE = 0, CANAL = 1, NUBE = 2, SUELO = 3 };

struct Malla {
    int nx, ny;
    std::vector<float> phi;           // idx = j*nx + i
    std::vector<std::uint8_t> tipo;
};

struct Resultado {
    long pasos = 0, iteraciones = 0;
    double t_laplace = 0, t_total = 0;
    std::vector<int> orden;  // paso en que entró cada celda al canal: -1 = no es canal, 0 = semilla
};

// Resuelve Laplace en las celdas de aire con rojo-negro + SOR, arrancando del φ
// que ya tenga la malla. Devuelve las iteraciones hechas.
using Solver = long (*)(Malla&, float omega, float tol);

Malla crear_malla(int L);
long solve_cpu(Malla& m, float omega, float tol);   // OpenMP
long solve_cuda(Malla& m, float omega, float tol);  // GPU
Resultado correr_rayo(Malla& m, Solver solve, float eta, unsigned semilla);
