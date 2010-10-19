all: train predict tbb-train

%:%.cpp
	$(CXX) -O2 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@ -ltbb

run: all
	./train 100 < 199/train > 199.model
	./predict 199.model < 199/test
