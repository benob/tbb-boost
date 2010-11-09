/*    This file is part of tbb-boost.

    tbb-boost is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    tbb-boost is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with tbb-boost.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

using namespace std;
using namespace tbb;

class Classifier {
public:
    string feature;
    double threshold;
    double** weight;
};

class ExampleProcessor {
    int num_labels;
    unordered_map<string, vector<Classifier> > const *classifiers;
    double const *default_score;
    vector<char*> *lines;
    int *output;
    //double *score;
public:
    ExampleProcessor(int num_labels, unordered_map<string, vector<Classifier> > const *classifiers, double const * default_score, vector<char*> *lines, int* output) {
        this->num_labels = num_labels;
        this->classifiers = classifiers;
        this->default_score = default_score;
        this->lines = lines;
        this->output = output;
        //this->score = new double[num_labels];
        //fprintf(stderr, "%p\n", this->score);
    }
    ~ExampleProcessor() {
        //delete this->score;
    }
    void operator() (const blocked_range<unsigned int>& range) const {
        //fprintf(stderr, "%d %d\n", range.begin(), range.end());
        for(unsigned int line = range.begin(); line != range.end(); line++) {
            char* save_pointer = NULL;
            char* token = strtok_r((*lines)[line], " \t:\n\r", &save_pointer);
            if(token == NULL || token[0] == '\0') {
                output[line] = -1;
                continue;
            }
            double score[num_labels];
            memcpy(score, default_score, sizeof(double) * num_labels);
            string feature;
            double value = 0;
            for(int i = 0; token != NULL; token = strtok_r(NULL, " \t:\n\r", &save_pointer), i++) {
                if(i == 0) {
                    // label
                } else if(i % 2 == 1) {
                    feature = token;
                } else {
                    value = strtod(token, NULL);
                    unordered_map<string, vector<Classifier> >::const_iterator found = classifiers->find(feature);
                    if(found != classifiers->end()) {
                        for(vector<Classifier>::const_iterator classifier = (*found).second.begin(); classifier != (*found).second.end(); classifier++) {
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
            output[line] = argmax;
        }
    }
};

int main(int argc, char** argv) {
    if(argc != 2) {
        fprintf(stdout, "USAGE: %s model < test > predictions\n", argv[0]);
        exit(1);
    }
    int num_labels = 0;
    unordered_map<string, vector<Classifier> > classifiers;
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
            if(fgets(buffer + strlen(buffer), buffer_size - strlen(buffer), stdin) == NULL) break;
        }
        char* token = strtok(buffer, " \t:\n\r");
        vector<const char*> values;
        for(;token != NULL; token = strtok(NULL, " \t:\n\r")) {
            values.push_back(token); //strtod(token, NULL));
        }
        if(state == 0) {
            labels.resize(values.size() / 2);
            for(int label = 0; label < (int) values.size(); label+= 2) {
                labels[strtol(values[label + 1], NULL, 10)] = values[label];
            }
            num_labels = labels.size();
            state = 1;
        } else if(state == 1) {
            classifier.feature = values[1];
            classifier.threshold = strtod(values[2], NULL);
            state = 2;
        } else if(state == 2) {
            classifier.weight = new double*[num_labels];
            for(int label = 0; label < num_labels; label++) {
                classifier.weight[label] = new double[3];
                classifier.weight[label][0] = strtod(values[label], NULL);
            }
            state = 3;
        } else if(state == 3) {
            for(int label = 0; label < num_labels; label++) classifier.weight[label][1] = strtod(values[label], NULL);
            state = 4;
        } else if(state == 4) {
            for(int label = 0; label < num_labels; label++) classifier.weight[label][2] = strtod(values[label], NULL);
            unordered_map<string, vector<Classifier> >::iterator found = classifiers.find(classifier.feature);
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
    for(unordered_map<string, vector<Classifier> >::iterator item = classifiers.begin(); item != classifiers.end(); item++) {
        for(vector<Classifier>::iterator classifier = (*item).second.begin(); classifier != (*item).second.end(); classifier++) {
            for(int label = 0; label < num_labels; label++) {
                default_score[label] += (*classifier).weight[label][0];
            }
        }
    }
    vector<char*> lines;
    while(NULL != fgets(buffer, buffer_size, stdin)) {
        while(buffer[strlen(buffer) - 1] != '\n') {
            buffer_size *= 2;
            buffer = (char*) realloc(buffer, buffer_size);
            if(fgets(buffer + strlen(buffer), buffer_size - strlen(buffer), stdin) == NULL) break;
        }
        lines.push_back(strdup(buffer));
    }
    free(buffer);
    int output[lines.size()];
    ExampleProcessor processor(num_labels, &classifiers, default_score, &lines, output);
    parallel_for(blocked_range<unsigned int>(0, lines.size()), processor, auto_partitioner());
    for(unsigned int line = 0 ; line < lines.size(); line++) {
        if(output[line] == -1) fprintf(stdout, "\n"); // pass empty lines as is
        else fprintf(stdout, "%s\n", labels[output[line]].c_str());
        free(lines[line]);
    }
    return 0;
}
