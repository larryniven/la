CXXFLAGS += -std=c++11 -I ../ -lblas
AR = gcc-ar

all: libla.a

clean:
	-rm libla.a
	-rm *.o

libla.a: la.o
	$(AR) rcs $@ $^

ifdef USE_GPU
la.o: la.cu
	nvcc -std=c++11 -c la.cu
endif
