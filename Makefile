CXXFLAGS += -std=c++11 -I ../
NVCCFLAGS += -std=c++11 -I ../
AR = gcc-ar

.PHONY: all clean gpu

all: libla.a

gpu: liblagpu.a

clean:
	-rm libla.a liblagpu.a
	-rm *.o

libla.a: la-cpu.o
	$(AR) rcs $@ $^

liblagpu.a: la-cpu.o la-gpu.o mem-pool.o
	$(AR) rcs $@ $^

la-gpu.o: la-gpu.cu
	nvcc $(NVCCFLAGS) -c la-gpu.cu

la-cpu.o: la-cpu.h la-cpu-impl.h
la-gpu.o: la-gpu.h la-gpu-impl.h
mem-pool.o: mem-pool.h
