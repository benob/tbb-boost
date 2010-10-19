all: train predict tbb-train

%:%.cpp
	$(CXX) -mtune=native -O2 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@
tbb-train: tbb-train.cpp
	$(CXX) -mtune=native -O2 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@ -ltbb

