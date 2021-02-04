// Weak Learners
#include "base_learner.hpp" // Class of Base Learner(NOT implemented)
#include "hypotheses/dstump.hpp" // Decision Stump
#include "hypotheses/bag_of_words.hpp" // Bag of Words

// C Libraries
#include <cmath>

// C++ Libraries
#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <set>

#ifndef __BOOSTER
#define __BOOSTER

class Booster
{
public:
    // ------------
    /* VARIABLES */
    std::string name; // name of boosting algorithm


    size_t sample_sz;  // # of training examples
    size_t feature_sz; // # of feature


    size_t max_iter; // Max iteration
    size_t terminated_iter; // Terminated iteration


    bool training_examples_exist;
    bool is_sparse;


    BaseLearner &base_learner;


    std::vector<std::vector<double> > dat; // A sequence of data
    std::vector<int> lab; // A sequence of label


    double cap; // Capping Parameter
    double tolerance; // Tolerance parameter


    double optimal_value; // Optimal Value


    // Combined classifiers with weights
    std::vector<double> weights;


    // Distribution over examples
    std::vector<double> dist;


    using hypothesis = std::function<int(std::vector<double>)>;
    std::vector<std::function<hypothesis> classifiers;

    // ------------



    // ---------------
    /* CONSTRUCTORS */
    Booster(void);


    Booster(const double tolerance_);


    Booster(const double tolerance_,
            const double cap_);


    // ---------------

    // ----------
    /* METHODS */

    void // Set training examples
    set_training_examples(const std::vector<std::vector<double> > &dat_,
                          const std::vector<int> &lab_);


    void
    set_sparsity(const bool is_sparse_);


    void
    set_cap(const double cap_);


    void
    set_base_learner(std::string name);


    void // Initialize variables for boosting process
    init_variables(void);


    std::function<int(std::vector<double>)> // Get weak hypothesis
    get_hypothesis(void);


    void
    to_edge_vector(const std::function<int(std::vector<double>)> &h);


    bool // Stopping criterion
    stopping_criterion(void);


    void // A post-processing that called only once
    terminate_process(void);


    void // Update parameters
    update_params(void);


    void
    update_distribution_over_examples(void);


    void // Main function that combines weak hypothesis
    boost(void);


    void
    debug(void);


    int // Predict label with combined hypothesis
    predict(const std::vector<double> &example);


    std::vector<int> // Predict labels with combined hypothesis
    predict(const std::vector<std::vector<double> > &examples);
    // ----------
};


inline
void
Booster::
set_sparsity(const bool is_sparse_)
{
    this->is_sparse = is_sparse_;
    return;
}


inline
void
Booster::
set_cap(const double cap_)
{
    this->cap = cap_;
}


inline
void
Booster::
debug()
{
    static int n = 1;
    std::cout << "debug, n = " << n++ << std::endl;
    return;
}

#endif
