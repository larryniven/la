CXXFLAGS += -std=c++11 -I ../
AR = gcc-ar

all: libla.a

clean:
	-rm libla.a liblagpu.a
	-rm *.o

libla.a: la.o
	$(AR) rcs $@ $^

liblagpu.a: la_gpu.o
	$(AR) rcs $@ $^

la_gpu.o: la_gpu.cu
	nvcc -std=c++11 -I ../ -c la_gpu.cu

la.o: la.h

