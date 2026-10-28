#include "dbm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

const int REVISAR_CADA = 10;
const long MAX_ITER = 1000000;

Malla crear_malla(int L) {
    Malla m{L, L, std::vector<float>(L * L, 1.0f), std::vector<std::uint8_t>(L * L, AIRE)};
    for (int i = 0; i < L; ++i) {
        m.phi[i] = 0, m.tipo[i] = NUBE;
        m.phi[(L - 1) * L + i] = 1, m.tipo[(L - 1) * L + i] = SUELO;
    }
    m.phi[L + L / 2] = 0, m.tipo[L + L / 2] = CANAL;  // semilla pegada a la nube
    return m;
}

// Promedio de las 4 vecinas; bordes laterales Neumann (la vecina que falta es la propia celda).
static inline float promedio(const float* p, int nx, int i, int j) {
    int il = i == 0 ? i : i - 1, ir = i == nx - 1 ? i : i + 1;
    return (p[j * nx + il] + p[j * nx + ir] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]) / 4.0f;
}

long solve_cpu(Malla& m, float omega, float tol) {
    const int nx = m.nx, ny = m.ny;
    float* p = m.phi.data();
    const std::uint8_t* t = m.tipo.data();
    for (long it = 1; it <= MAX_ITER; ++it) {
        for (int color = 0; color < 2; ++color) {
#pragma omp parallel for schedule(static)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = (j + color) % 2; i < nx; i += 2) {
                    int k = j * nx + i;
                    if (t[k] != AIRE) continue;
                    // El recorte evita que la sobre-relajación saque φ de [0, 1].
                    p[k] = std::clamp((1.0f - omega) * p[k] + omega * promedio(p, nx, i, j), 0.0f, 1.0f);
                }
        }
        if (it % REVISAR_CADA) continue;
        float res = 0;
#pragma omp parallel for schedule(static) reduction(max : res)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 0; i < nx; ++i)
                if (t[j * nx + i] == AIRE) res = std::max(res, std::fabs(promedio(p, nx, i, j) - p[j * nx + i]));
        if (res < tol) return it;
    }
    throw std::runtime_error("solve_cpu: no convergió");
}

// Hace crecer el rayo celda por celda hasta que toca el suelo. Cada paso:
//   1. resuelve Laplace con el canal actual (φ del aire),
//   2. busca las candidatas: celdas de aire pegadas al canal,
//   3. sortea una con probabilidad φ^η / Σ φ^η,
//   4. la agrega al canal.
// `solve` es solve_cpu o solve_cuda: es lo único que cambia entre las dos versiones.
Resultado correr_rayo(Malla& m, Solver solve, float eta, unsigned semilla) {
    using reloj = std::chrono::steady_clock;
    const int nx = m.nx, ny = m.ny;
    const float omega = 2.0f / (1.0f + std::sin(3.14159265f / nx));  // ω óptimo de SOR
    const float tol = 1e-5f;
    std::mt19937 rng(semilla); 
    Resultado r;

    // orden[k] = paso en que la celda k entró al canal (-1 = nunca). La semilla es el paso 0.
    r.orden.assign(nx * ny, -1);
    for (int k = 0; k < nx * ny; ++k)
        if (m.tipo[k] == CANAL) r.orden[k] = 0;

    auto t0 = reloj::now();
    std::vector<int> cand;      // índices de las candidatas
    std::vector<double> acum;   // acum[c] = suma de los pesos de cand[0..c]
    for (;;) {
        // 1. Laplace. φ no se reinicia: parte de la solución del paso anterior,
        //    que ya está casi lista, y bastan pocas iteraciones.
        auto t1 = reloj::now();
        r.iteraciones += solve(m, omega, tol);
        r.t_laplace += std::chrono::duration<double>(reloj::now() - t1).count();

        // 2. Candidatas: celdas de aire con alguna de sus 4 vecinas en el canal.
        //    Se recorre toda la malla en cada paso; es barato comparado con Laplace.
        //    Las filas 0 (nube) y ny-1 (suelo) nunca son aire, por eso j va de 1 a ny-2
        //    y k - nx, k + nx siempre existen. A los lados se revisa no salirse de la malla.
        cand.clear(), acum.clear();
        double S = 0;  // suma total de los pesos
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 0; i < nx; ++i) {
                int k = j * nx + i;
                if (m.tipo[k] != AIRE) continue;
                if (m.tipo[k - nx] == CANAL || m.tipo[k + nx] == CANAL ||
                    (i > 0 && m.tipo[k - 1] == CANAL) || (i < nx - 1 && m.tipo[k + 1] == CANAL)) {
                    // Peso φ^η: el rayo prefiere avanzar donde el potencial (el campo) es mayor.
                    S += std::pow((double)m.phi[k], (double)eta);
                    cand.push_back(k), acum.push_back(S);
                }
            }

        // 3. Sorteo por suma acumulada. Cada candidata ocupa en [0, S) un tramo tan largo
        //    como su peso; x cae en el tramo de cand[c] con probabilidad peso / S.
        //    upper_bound da la primera candidata con acum > x, la dueña de ese tramo.
        double x = std::uniform_real_distribution<double>(0, S)(rng);
        int k = cand[std::upper_bound(acum.begin(), acum.end(), x) - acum.begin()];

        // 4. La celda elegida pasa a ser canal, con el potencial de la nube (φ = 0).
        //    En el siguiente paso Laplace la toma como borde fijo.
        m.tipo[k] = CANAL, m.phi[k] = 0;
        r.orden[k] = ++r.pasos;
        if (m.tipo[k + nx] == SUELO) break;  // el suelo solo está en la última fila
    }
    r.t_total = std::chrono::duration<double>(reloj::now() - t0).count();
    return r;
}
