CXXFLAGS += -std=c++11 -I ../ -lblas
AR = gcc-ar

all: libla.a

clean:
	-rm libla.a
	-rm *.o

libla.a: la.o
	$(AR) rcs $@ $^
