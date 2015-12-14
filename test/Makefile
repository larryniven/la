CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt

.PHONY: all clean

all: test-la

clean:
	-rm test-la
	-rm *.o

test-la: test-la.o
	$(CXX) $(CXXFLAGS) -o $@ $^ ../la.o -lebt -lcblas
