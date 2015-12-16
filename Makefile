CXXFLAGS += -std=c++11 -I ../
AR = gcc-ar

.PHONY: all clean gpu

all: libla.a

gpu: liblagpu.a

clean:
	-rm libla.a liblagpu.a
	-rm *.o

libla.a: la.o
	$(AR) rcs $@ $^

liblagpu.a: la-gpu.o
	$(AR) rcs $@ $^

la-gpu.o: la-gpu.cu
	nvcc $(CXXFLAGS) -c la-gpu.cu

la.o: la.h
la-gpu.o: la-gpu.h
