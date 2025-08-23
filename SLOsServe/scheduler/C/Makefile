CXX = g++
CXXFLAGS = -I.  # Include debug symbols and current directory for headers

all: test test_multi_tpot

test: test.cc promax_spec_sch.cc
	$(CXX) $(CXXFLAGS) -g -o $@ $^

# test_multi_tpot: test.cc promax_spec_multi_tpot.cc
# 	$(CXX) $(CXXFLAGS) -o $@ $^

err: err.cc promax_spec_sch.cc
	$(CXX) $(CXXFLAGS) -g -o $@ $^

timer: timer.cc promax_spec_sch.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm test err