#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <bitset>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"

using namespace std;
using namespace tbb;

class Example {
public:
    int32_t label;
    double* weight;
    double* score;
};

class Feature {
public:
    int32_t id;
    const char* name;
    vector< pair<int32_t, double> > index;
};

bool comparator(const pair<int32_t, double>& a, const pair<int32_t, double>& b)
{
      return a.second < b.second;
}

double compute_objective(double weight[][3][2], int num_labels) {
    double output = 0;
    for(int label = 0; label < num_labels; label++) {
        double w0 = weight[label][0][1] * weight[label][0][0]; if(w0 > 0) output += sqrt(w0);
        double w1 = weight[label][1][1] * weight[label][1][0]; if(w1 > 0) output += sqrt(w1);
        double w2 = weight[label][2][1] * weight[label][2][0]; if(w2 > 0) output += sqrt(w2);
    }
    return output * 2;
}

class MinimizeObjective {
    vector<Example> examples;
    vector<Feature> features;
    vector<bool> is_known;
    int num_labels;
public:
    double min;
    int32_t argmin;
    double argmin_threshold;
    double (*argmin_weight)[3][2];
    MinimizeObjective(int num_labels, vector<Example> &examples, vector<Feature> &features) {
        this->num_labels = num_labels;
        this->examples = examples;
        this->features = features;
        argmin_weight = new double[num_labels][3][2];
        min = 0;
        argmin = -1;
        argmin_threshold = -1;
    }
    MinimizeObjective(MinimizeObjective& other, split) { 
        examples = other.examples;
        features = other.features;
        num_labels = other.num_labels;
        argmin_weight = new double[num_labels][3][2];
        memcpy(argmin_weight, other.argmin_weight, sizeof(double) * num_labels * 3 * 2);
        min = other.min;
        argmin = other.argmin;
        argmin_threshold = other.argmin_threshold;
    }
    void operator() (const blocked_range<int>& feature_range) {
        for(int id = feature_range.begin(); id != feature_range.end(); id++) {
            Feature feature = features[id];

            //memset(is_known, 0, sizeof(is_known));
            is_known.assign(examples.size(), false);
            //unordered_set<int32_t> is_known;
            for(vector< pair<int32_t, double> >::iterator value = feature.index.begin(); value != feature.index.end(); value++) {
                //is_known.insert((*value).first);
                is_known[(*value).first] = true;
            }
            double weight[num_labels][3][2];
            for(int label = 0; label < num_labels; label++) {
                weight[label][0][0] = weight[label][0][1] = weight[label][1][0] = weight[label][1][1] = weight[label][2][0] = weight[label][2][1] = 0;
            }
            // initialize weights
            for(int32_t i = 0; i < (int32_t) examples.size(); i++) {
                //if(is_known.find(i) != is_known.end()) {
                Example example = examples[i];
                for(int label = 0; label < num_labels; label++) {
                    if(is_known[i]) {
                        if(label == example.label) weight[label][2][1] += example.weight[label];
                        else weight[label][2][0] += example.weight[label];
                    } else {
                        if(label == example.label) weight[label][0][1] += example.weight[label];
                        else weight[label][0][0] += example.weight[label];
                    }
                }
            }
            double objective = compute_objective(weight, num_labels);
            //fprintf(stderr, "OBJ: %d %g %g\n", id, -DBL_MAX, objective);
            if(objective < min || argmin == -1) {
                min = objective;
                argmin = id;
                argmin_threshold = -DBL_MAX;
                memcpy(argmin_weight, weight, sizeof(double) * num_labels * 3 * 2);
            }
            // try all possible thresholds
            double previous_value = (*feature.index.begin()).second;
            for(vector< pair<int32_t, double> >::iterator value = feature.index.begin(); value != feature.index.end(); value++) {
                if((*value).second > previous_value) {
                    double objective = compute_objective(weight, num_labels);
                    //fprintf(stderr, "OBJ: %d %g %g\n", id, (*value).second, objective);
                    if(objective < min || argmin == -1) {
                        min = objective;
                        argmin = id;
                        argmin_threshold = ((*value).second + previous_value) / 2;
                        memcpy(argmin_weight, weight, sizeof(double) * num_labels * 2 * 3);
                    }
                }
                Example example = examples[(*value).first];
                for(int label = 0; label < num_labels; label++) {
                    if(label == example.label) weight[label][1][1] += example.weight[label];
                    else weight[label][1][0] += example.weight[label];
                    if(label == example.label) weight[label][2][1] -= example.weight[label];
                    else weight[label][2][0] -= example.weight[label];
                }
                previous_value = (*value).second;
            }
        }
        //fprintf(stderr, "RANGE: %d->%d (%d), min=%g argmin=%d\n", feature_range.begin(), feature_range.end(), features.size(), min, argmin);
    }
    void join(const MinimizeObjective& other) {
        if(other.min < min || argmin == -1) {
            min = other.min;
            argmin = other.argmin;
            argmin_threshold = other.argmin_threshold;
            memcpy(argmin_weight, other.argmin_weight, num_labels * 3 * 2 * sizeof(double));
        }
    }

    ~MinimizeObjective() {
        delete argmin_weight;
    }
};

int main(int argc, char** argv) {
    if(argc != 2) {
        fprintf(stdout, "USAGE: %s iterations < train > model\n", argv[0]);
        exit(1);
    }
    vector<Example> examples;
    unordered_map<string, int> feature_map;
    vector<Feature> features;
    unordered_map<string, int> labels;
    int num_labels = 0;
    int buffer_size = 1024;
    char* buffer = (char*) malloc(buffer_size);
    // read examples
    while(NULL != fgets(buffer, buffer_size, stdin)) {
        while(buffer[strlen(buffer) - 1] != '\n') {
            buffer_size *= 2;
            buffer = (char*) realloc(buffer, buffer_size);
            fgets(buffer + strlen(buffer), buffer_size - strlen(buffer), stdin);
        }
        char* token = strtok(buffer, " \t:\n\r");
        int i;
        int label = 0;
        string name;
        double value;
        for(i = 0; token != NULL; token = strtok(NULL, " \t:\n\r"), i++) {
            if(i == 0) {
                string label_str(token);
                unordered_map<string, int>::iterator found = labels.find(label_str);
                if(found == labels.end()) {
                    label = (int) labels.size();
                    labels[label_str] = label;
                } else {
                    label = (*found).second;
                }
                if(label >= num_labels) num_labels = label + 1;
            } else if(i % 2 == 1) {
                name = token;
            } else {
                value = strtod(token, NULL);
                unordered_map<string, int>::iterator found = feature_map.find(name);
                if(found == feature_map.end()) {
                    Feature feature;
                    feature.id = features.size();
                    feature.name = strdup(name.c_str());
                    feature.index.push_back(pair<int32_t, double>(examples.size(), value));
                    features.push_back(feature);
                    feature_map[name] = feature.id;
                } else {
                    features[found->second].index.push_back(pair<int32_t, double>(examples.size(), value));
                }
            }
        }
        Example example;
        example.label = label;
        examples.push_back(example);
    }
    free(buffer);
    // initialize weights
    for(vector<Example>::iterator it = examples.begin(); it != examples.end(); it++) {
        (*it).weight = new double[num_labels];
        for(int i = 0; i < num_labels; i++) (*it).weight[i] = 1.0 / (examples.size() * num_labels);
        (*it).score = new double[num_labels];
        for(int i = 0; i < num_labels; i++) (*it).score[i] = 0;
    }

    // sort feature values
    for(vector<Feature>::iterator item = features.begin(); item != features.end(); item++) {
        sort((*item).index.begin(), (*item).index.end(), comparator);
    }

    for(unordered_map<string, int>::iterator label = labels.begin(); label != labels.end(); label++) {
        if(label != labels.begin()) fprintf(stdout, " ");
        fprintf(stdout, "%s:%d", (*label).first.c_str(), (*label).second);
    }
    fprintf(stdout, "\n");
    int num_iterations = strtol(argv[1], NULL, 10);
    fprintf(stderr, "examples:%zd features:%zd labels:%d iterations:%d\n", examples.size(), features.size(), num_labels, num_iterations);
    for(int iteration = 0; iteration < num_iterations; iteration++) {
        MinimizeObjective minimizer(num_labels, examples, features);
        parallel_reduce(blocked_range<int>(0, features.size(), 1), minimizer, auto_partitioner());
        double min = minimizer.min;
        int32_t argmin = minimizer.argmin;
        double argmin_threshold = minimizer.argmin_threshold;
        double argmin_weight[num_labels][3][2];
        memcpy(argmin_weight, minimizer.argmin_weight, num_labels * 3 * 2 * sizeof(double));

        /*double min = 0;
        int32_t argmin = -1;
        double argmin_threshold = -1;
        double argmin_weight[num_labels][3][2] ;
        int num = 0;
        //bool is_known[examples.size()];
        vector<bool> is_known(examples.size());
        for(vector<Feature>::iterator item = features.begin(); item != features.end(); item++) {
            //memset(is_known, 0, sizeof(is_known));
            is_known.assign(examples.size(), false);
            //unordered_set<int32_t> is_known;
            for(vector< pair<int32_t, double> >::iterator value = (*item).index.begin(); value != (*item).index.end(); value++) {
                //is_known.insert((*value).first);
                is_known[(*value).first] = true;
            }
            double weight[num_labels][3][2];
            for(int label = 0; label < num_labels; label++) {
                weight[label][0][0] = weight[label][0][1] = weight[label][1][0] = weight[label][1][1] = weight[label][2][0] = weight[label][2][1] = 0;
            }
            // initialize weights
            for(int32_t i = 0; i < (int32_t) examples.size(); i++) {
                //if(is_known.find(i) != is_known.end()) {
                Example example = examples[i];
                for(int label = 0; label < num_labels; label++) {
                    if(is_known[i]) {
                        if(label == example.label) weight[label][2][1] += example.weight[label];
                        else weight[label][2][0] += example.weight[label];
                    } else {
                        if(label == example.label) weight[label][0][1] += example.weight[label];
                        else weight[label][0][0] += example.weight[label];
                    }
                }
            }
            double objective = compute_objective(weight, num_labels);
            //fprintf(stderr, "OBJ: %d %g %g\n", (*item).id, -DBL_MAX, objective);
            if(objective < min || argmin == -1) {
                min = objective;
                argmin = (*item).id;
                argmin_threshold = -DBL_MAX;
                memcpy(argmin_weight, weight, sizeof(argmin_weight));
            }
            // try all possible thresholds
            double previous_value = (*(*item).index.begin()).second;
            for(vector< pair<int32_t, double> >::iterator value = (*item).index.begin(); value != (*item).index.end(); value++) {
                if((*value).second > previous_value) {
                    double objective = compute_objective(weight, num_labels);
                    //fprintf(stderr, "OBJ: %d %g %g\n", (*item).id, (*value).second, objective);
                    if(objective < min || argmin == -1) {
                        min = objective;
                        argmin = (*item).id;
                        argmin_threshold = ((*value).second + previous_value) / 2;
                        memcpy(argmin_weight, weight, sizeof(argmin_weight));
                    }
                }
                Example example = examples[(*value).first];
                for(int label = 0; label < num_labels; label++) {
                    if(label == example.label) weight[label][1][1] += example.weight[label];
                    else weight[label][1][0] += example.weight[label];
                    if(label == example.label) weight[label][2][1] -= example.weight[label];
                    else weight[label][2][0] -= example.weight[label];
                }
                previous_value = (*value).second;
            }
            num++;
        }*/
        fprintf(stdout, "%d %d %g %g\n", iteration, argmin, argmin_threshold, min);
        fprintf(stderr, "iteration:%d feature:%d threshold:%g min-objective:%g\n", iteration, argmin, argmin_threshold, min);
        // compute classifier weights
        double classifier[num_labels][3];
        double epsilon = 0.5 / (num_labels * examples.size());
        for(int label = 0; label < num_labels; label++) {
            classifier[label][0] = 0.5 * log((argmin_weight[label][0][1] + epsilon) / (argmin_weight[label][0][0] + epsilon));
            classifier[label][1] = 0.5 * log((argmin_weight[label][1][1] + epsilon) / (argmin_weight[label][1][0] + epsilon));
            classifier[label][2] = 0.5 * log((argmin_weight[label][2][1] + epsilon) / (argmin_weight[label][2][0] + epsilon));
        }
        for(int label = 0; label < num_labels; label ++) fprintf(stdout, "%g ", classifier[label][0]); fprintf(stdout, "\n");
        for(int label = 0; label < num_labels; label ++) fprintf(stdout, "%g ", classifier[label][1]); fprintf(stdout, "\n");
        for(int label = 0; label < num_labels; label ++) fprintf(stdout, "%g ", classifier[label][2]); fprintf(stdout, "\n");
        fprintf(stdout, "\n");
        // apply classifier and update weights
        Feature feature = features[argmin];
        unordered_set<int32_t> known_value;
        for(vector< pair<int32_t, double> >::iterator value = feature.index.begin(); value != feature.index.end(); value++) {
            known_value.insert((*value).first);
            Example example = examples[(*value).first];
            if((*value).second < argmin_threshold) {
                for(int label = 0; label < num_labels; label++) {
                    example.score[label] += classifier[label][1];
                    if(fabs(classifier[label][1]) > 1e-11) {
                        if(label == example.label) example.weight[label] *= exp(-classifier[label][1]);
                        else example.weight[label] *= exp(classifier[label][1]);
                    }
                }
            } else {
                for(int label = 0; label < num_labels; label++) {
                    example.score[label] += classifier[label][2];
                    if(fabs(classifier[label][2]) > 1e-11) {
                        if(label == example.label) example.weight[label] *= exp(-classifier[label][2]);
                        else example.weight[label] *= exp(classifier[label][2]);
                    }
                }
            }
        }
        double norm = 0;
        for(int i = 0; i < (int) examples.size(); i++) {
            Example example = examples[i];
            if(known_value.find(i) == known_value.end()) {
                for(int label = 0; label < num_labels; label++) {
                    example.score[label] += classifier[label][0];
                    if(fabs(classifier[label][0]) > 1e-11) {
                        if(label == example.label) example.weight[label] *= exp(-classifier[label][0]);
                        else example.weight[label] *= exp(classifier[label][0]);
                    }
                }
            }
            for(int label = 0; label < num_labels; label++) norm += example.weight[label];
        }
        for(int i = 0; i < (int) examples.size(); i++) {
            Example example = examples[i];
            for(int label = 0; label < num_labels; label++) example.weight[label] /= norm;
        }
        // update weights
    }
    return 0;
}
