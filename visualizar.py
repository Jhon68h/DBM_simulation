"""Anima el crecimiento del rayo a partir de salida/orden.txt.

Uso: python visualizar.py [salida/orden.txt] [fotogramas=150]
Guarda rayo.gif junto al archivo; cada fotograma muestra las celdas con paso <= t.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ruta = Path(sys.argv[1] if len(sys.argv) > 1 else "salida/orden.txt")
fotogramas = int(sys.argv[2]) if len(sys.argv) > 2 else 150
orden = np.loadtxt(ruta, dtype=int)
pasos = orden.max()
tiempos = np.unique(np.linspace(0, pasos, fotogramas).astype(int))

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor("black")
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(np.ma.masked_less(orden, 0), cmap="plasma", vmin=0, vmax=pasos, interpolation="nearest")
fig.colorbar(im, ax=ax, label="paso de crecimiento", shrink=0.8)


def dibujar(t):
    im.set_data(np.ma.masked_where((orden < 0) | (orden > t), orden))
    ax.set_title(f"Rayo DBM {orden.shape[1]}×{orden.shape[0]}  paso {t}/{pasos}")
    return (im,)


anim = FuncAnimation(fig, dibujar, frames=tiempos, interval=40)
salida = ruta.with_name("rayo.gif")
anim.save(salida, writer=PillowWriter(fps=25))
print(f"guardado {salida} ({len(tiempos)} fotogramas)")
