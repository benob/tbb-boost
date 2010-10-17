all: train predict

%:%.cpp
	$(CXX) -O2 -g -std=c++0x -Wall $(CXXFLAGS) $< -o $@

run: all
	./train 100 < 199/train > 199.model
	./predict 199.model < 199/test
