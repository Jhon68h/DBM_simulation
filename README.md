# Simulación de rayos con el Modelo de Ruptura Dieléctrica (DBM)

Simula en 2D cómo crece un rayo desde una nube hasta el suelo y compara cuánto se
acelera el cálculo al paralelizarlo en CPU (OpenMP) y en GPU (CUDA).

## El modelo

Malla de L×L celdas: arriba la nube (φ = 0), abajo el suelo (φ = 1) y el canal
empieza pegado a la nube en la columna central (φ = 0). En cada paso:

1. Se resuelve la ecuación de Laplace (∇²φ = 0) en el aire.
2. Se buscan las candidatas: celdas de aire vecinas del canal.
3. Se elige una con probabilidad proporcional a φ^η.
4. Se agrega al canal; si toca el suelo, el rayo termina.

η = 0 da una mancha compacta, η = 1 un rayo ramificado y η = 3 uno casi recto.

## Archivos

| Archivo | Contenido |
|---|---|
| `dbm.hpp`, `dbm.cpp` | Malla, solver OpenMP, candidatas, sorteo y bucle del rayo |
| `laplace.cu` | Solver en GPU con CUDA |
| `main.cpp` | Corre el rayo en CPU y GPU, compara y guarda la salida |
| `visualizar.py` | Animación del crecimiento (requiere matplotlib) |
| `poster.pdf` | Póster con los resultados |

## Uso

```
make
OMP_NUM_THREADS=6 ./dbm 256 1 2201492     # L eta semilla
python visualizar.py
```

Imprime pasos, iteraciones, tiempos y speedup, y termina con código 1 si la CPU y
la GPU no hicieron el mismo rayo. Escribe en `salida/` las imágenes `cpu.pgm` y
`cuda.pgm`, `orden.txt` (paso en que cada celda entró al canal) y `rayo.gif`.

## Paralelización

Solo se paraleliza el solver de Laplace (99,2 % del tiempo); el bucle del rayo es
secuencial porque cada paso depende del anterior.

- **Rojo-negro con SOR:** la malla se colorea como un tablero de ajedrez; las
  vecinas de una celda roja son negras y al revés, así que cada color se actualiza
  entero en paralelo. ω = 2 / (1 + sin(π/L)) ≈ 1,976 con L = 256. Converge 26
  veces más rápido que Jacobi, y Gauss-Seidel no se puede paralelizar.
- **OpenMP:** `#pragma omp parallel for schedule(static)` sobre las filas de cada
  color y `reduction(max:res)` para el residuo.
- **CUDA:** un hilo por celda en bloques de 32×8, un kernel por color y
  `atomicMax` sobre el float leído como entero para el residuo.
- El residuo se revisa cada 10 iteraciones y φ no se reinicia entre pasos
  (~118 iteraciones por paso).

## Resultados

Ryzen 5 4600H y GTX 1650 Ti, η = 1, semilla 2201492, mediana de 5 corridas:

| L | T_ref, 1 hilo (s) | T_omp, 6 hilos (s) | S_omp | T_cuda (s) | S_cuda |
|---|---|---|---|---|---|
| 64 | 0,267 | 0,156 | 1,7× | 0,160 | 1,7× |
| 128 | 4,93 | 1,81 | 2,7× | 0,919 | 5,4× |
| 192 | 28,4 | 9,33 | 3,0× | 2,94 | 9,7× |
| 256 | 84,4 | 27,0 | 3,1× | 6,36 | 13,3× |

## Verificación

La CPU y la GPU producen el mismo rayo bit a bit: rojo-negro da el mismo φ con
cualquier número de hilos, el residuo es un máximo (no depende del orden), el
sorteo usa el mismo `std::mt19937` y la GPU se compila con `--fmad=false` para
redondear igual que la CPU.
