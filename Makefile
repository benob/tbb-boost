all: train predict tbb-train tbb-predict

%:%.cpp
	$(CXX) -mtune=native -O3 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@
tbb-train: tbb-train.cpp
	$(CXX) -mtune=native -O3 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@ -ltbb
tbb-predict: tbb-predict.cpp
	$(CXX) -mtune=native -O3 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@ -ltbb

