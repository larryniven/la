CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt
NVCCFLAGS += -std=c++11 -I ../../ -L ../../ebt

.PHONY: all clean

all: test-la test-mem-pool

gpu: test-la-gpu

clean:
	-rm test-la test-la-gpu
	-rm *.o

test-la: test-la.o
	$(CXX) $(CXXFLAGS) -o $@ $^ ../la-cpu.o -lebt -lblas

test-la-gpu.o: test-la-gpu.cu
	nvcc $(NVCCFLAGS) -c $^

test-la-gpu: test-la-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ ../la-cpu.o ../la-gpu.o $^ -lblas -lcublas -lcudart -lebt

test-mem-pool: ../mem-pool.o test-mem-pool.o
	$(CXX) $(CXXFLAGS) -o $@ $^
