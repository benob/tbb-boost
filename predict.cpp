#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

class Classifier {
public:
    int feature;
    double threshold;
    double** weight;
};

int main(int argc, char** argv) {
    if(argc != 2) {
        fprintf(stdout, "USAGE: %s model < test > predictions\n", argv[0]);
        exit(1);
    }
    int num_labels = 0;
    unordered_map<int, vector<Classifier> > classifiers;
    vector<string> labels;
    FILE* model = fopen(argv[1], "r");
    if(model == NULL) {
        perror("Error loading model");
        exit(2);
    }
    int buffer_size = 1024;
    char* buffer = (char*) malloc(buffer_size);
    // read model
    Classifier classifier;
    int state = 0;
    while(NULL != fgets(buffer, buffer_size, model)) {
        while(buffer[strlen(buffer) - 1] != '\n') {
            buffer_size *= 2;
            buffer = (char*) realloc(buffer, buffer_size);
            fgets(buffer + strlen(buffer), buffer_size - strlen(buffer), stdin);
        }
        char* token = strtok(buffer, " \t:\n\r");
        vector<double> values;
        for(;token != NULL; token = strtok(NULL, " \t:\n\r")) {
            values.push_back(strtod(token, NULL));
        }
        if(state == 0) {
            for(int label = 0; label < values.size(); label++) {
                labels.push_back(string(values[label]));
            }
            state = 1;
        } else if(state == 1) {
            classifier.feature = (int) values[1];
            classifier.threshold = values[2];
            state = 2;
        } else if(state == 2) {
            if(num_labels < (int) values.size()) num_labels = values.size();
            classifier.weight = new double*[num_labels];
            for(int label = 0; label < num_labels; label++) {
                classifier.weight[label] = new double[3];
                classifier.weight[label][0] = values[label];
            }
            state = 3;
        } else if(state == 3) {
            for(int label = 0; label < num_labels; label++) classifier.weight[label][1] = values[label];
            state = 4;
        } else if(state == 4) {
            for(int label = 0; label < num_labels; label++) classifier.weight[label][2] = values[label];
            unordered_map<int, vector<Classifier> >::iterator found = classifiers.find(classifier.feature);
            if(found == classifiers.end()) {
                classifiers[classifier.feature] = vector<Classifier>();
            }
            classifiers[classifier.feature].push_back(classifier);
            state = 5;
        } else if(state == 5) {
            state = 1;
        }
    }
    fclose(model);
    double default_score[num_labels];
    for(unordered_map<int, vector<Classifier> >::iterator item = classifiers.begin(); item != classifiers.end(); item++) {
        for(vector<Classifier>::iterator classifier = (*item).second.begin(); classifier != (*item).second.end(); classifier++) {
            for(int label = 0; label < num_labels; label++) {
                default_score[label] += (*classifier).weight[label][0];
            }
        }
    }
    while(NULL != fgets(buffer, buffer_size, stdin)) {
        while(buffer[strlen(buffer) - 1] != '\n') {
            buffer_size *= 2;
            buffer = (char*) realloc(buffer, buffer_size);
            fgets(buffer + strlen(buffer), buffer_size - strlen(buffer), stdin);
        }
        char* token = strtok(buffer, " \t:\n\r");
        double score[num_labels];
        memcpy(score, default_score, sizeof(score));
        int feature = 0;
        double value = 0;
        for(int i = 0; token != NULL; token = strtok(NULL, " \t:\n\r"), i++) {
            if(i == 0) {
                // label
            } else if(i % 2 == 1) {
                feature = strtol(token, NULL, 10);
            } else {
                value = strtod(token, NULL);
                unordered_map<int, vector<Classifier> >::iterator found = classifiers.find(feature);
                if(found != classifiers.end()) {
                    for(vector<Classifier>::iterator classifier = (*found).second.begin(); classifier != (*found).second.end(); classifier++) {
                        for(int label = 0; label < num_labels; label++) {
                            score[label] -= (*classifier).weight[label][0];
                            if(value < (*classifier).threshold) score[label] += (*classifier).weight[label][1];
                            else score[label] += (*classifier).weight[label][2];
                        }
                    }
                }
            }
        }
        double max = 0;
        int argmax = -1;
        for(int label = 0; label < num_labels; label++) {
            if(max < score[label] || argmax == -1) {
                argmax = label;
                max = score[label];
            }
        }
        fprintf(stdout, "%d\n", argmax + 1);
    }
    free(buffer);
    return 0;
}