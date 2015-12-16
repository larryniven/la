CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt

.PHONY: all clean

all: test-la

gpu: test-la-gpu

clean:
	-rm test-la test-la-gpu
	-rm *.o

test-la: test-la.o
	$(CXX) $(CXXFLAGS) -o $@ $^ ../la.o -lebt -lcblas

test-la-gpu.o: test-la-gpu.cu
	nvcc $(CXXFLAGS) -c $^

test-la-gpu: test-la-gpu.o
	nvcc $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ ../la-gpu.o $^ -lcublas -lcudart -lebt
