all: train predict tbb-train tbb-predict

CXXFLAGS+=-mtune=native -O3 -g -std=c++0x -Wall

%:%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
tbb-train: tbb-train.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -ltbb
tbb-predict: tbb-predict.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -ltbb

test: all a9a.model1 a9a.model2 a9a.test
	./predict a9a.model1 < a9a.test | paste - a9a.test | awk '{if($$1!=$$2){e++}n++}END{print e,n,e/n}'
	./tbb-predict a9a.model2 < a9a.test | paste - a9a.test | awk '{if($$1!=$$2){e++}n++}END{print e,n,e/n}'
a9a.model1 a9a.model2: a9a.train
	./train 10 < a9a.train > a9a.model1
	./tbb-train 10 < a9a.train > a9a.model2
a9a.train:
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a -O a9a.train
a9a.test:
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t -O a9a.test

clean:
	rm -f train predict tbb-train tbb-predict a9a.model1 a9a.model2 a9a.test a9a.train
