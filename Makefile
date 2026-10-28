NVCC ?= /opt/cuda/bin/nvcc

# --fmad=false: sin FMA la GPU redondea igual que la CPU y ambos hacen el mismo rayo.
dbm: main.cpp dbm.cpp laplace.cu dbm.hpp
	$(NVCC) -O2 -std=c++17 -arch=sm_75 --fmad=false -Xcompiler -fopenmp,-Wall main.cpp dbm.cpp laplace.cu -o dbm

clean:
	rm -f dbm
