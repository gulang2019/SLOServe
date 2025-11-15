CXX = g++
CXXFLAGS = -I.  # Include debug symbols and current directory for headers

PYBIND_FLAGS = -fPIC $(python3 -m pybind11 --includes) -std=c++17 

SRCS = $(wildcard *.cc)


all: test

main: $(SRCS)
	$(CXX) $(CXXFLAGS) -g -o $@ $^

test: test.cc promax_spec_sch.cc
	$(CXX) $(CXXFLAGS) -g -o $@ $^
promax: bind.cc promax_spec_sch.cc
	$(CXX) -O3 -Wall -shared -std=c++17 -fPIC $(shell python3 -m pybind11 --includes) -I. -o promax$(shell python3-config --extension-suffix) bind.cc promax_spec_sch.cc

# test_multi_tpot: test.cc promax_spec_multi_tpot.cc
# 	$(CXX) $(CXXFLAGS) -o $@ $^

err: err.cc promax_spec_sch.cc
	$(CXX) $(CXXFLAGS) -g -o $@ $^

timer: timer.cc promax_spec_sch.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm test err
