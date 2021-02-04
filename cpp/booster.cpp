#include "base_booster.hpp"

// ---------------
/* CONSTRUCTORS */
Booster::
Booster(void)
{
    this->name = "";

    this->sample_sz  = 0;
    this->feature_sz = 0;

    this->max_iter = 0;
    this->terminated_iter = 0;

    this->cap = 1.0;

    this->training_examples_exist = false;
    this->is_sparse = false;
}


Booster::
Booster(const double tolerance_)
    : tolerance(tolerance_)
{
    this->name = "";

    this->sample_sz  = 0;
    this->feature_sz = 0;

    this->max_iter = 0;
    this->terminated_iter = 0;

    this->cap = 1.0;

    this->training_examples_exist = false;
    this->is_sparse = false;
}


Booster::
Booster(const double tolerance_,
        const double cap_)
    : tolerance(tolerance_),
      cap(cap_)
{
    this->name = "";

    this->sample_sz  = 0;
    this->feature_sz = 0;

    this->max_iter = 0;
    this->terminated_iter = 0;

    this->training_examples_exist = false;
    this->is_sparse = false;
}


// ---------------


// ----------
/* METHODS */


void
Booster::
set_base_learner(std::string name)
{
    std::transform(name.begin(), name.end(),
                   name.begin(), ::tolower);


    if (name == "dstump" || name == "decisionstump")
        base_learner = dstump::DStump(this->dat, this->lab);


    return;
}


void
Booster::
set_training_examples(const std::vector<std::vector<double> > &dat_,
                      const std::vector<int> &lab_)
    : dat(dat_),
      lab(lab_)
{
    this->sample_sz  = this->dat.size();
    this->feature_sz = this->dat[0].size();

    this->training_examples_exist = true;
    this->is_sparse = false;
    return;
}


void
Booster::
init_variables(void)
{
    this->dist.resize(this->sample_sz);
    std::fill(this->dist.begin(),
              this->dist.end(),
              1.0 / (double) this->m);


    this->weights.resize(0);


    this->classifiers.resize(0);
    return;
}


std::function<int(std::vector<double>)>
Booster::
get_hypothesis(void)
{
    auto h = this->base_learner.get_hypothesis(this->dist);
    return h;
}


void
Booster::
boost(void)
{
    if (!this->training_examples_exist)
    {
        std::cerr << "Error: No training examples specified."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }


    // Initialize variables
    this->init_variables();


    std::cout << "Running " << this->name
              << " ... " << std::flush;

    for (size_t iter = 1; iter < this->max_iter; ++iter)
    {
        // Get weak hypothesis
        this->get_hypothesis();


        // Check stopping criterion
        if (this->stopping_criterion())
        {
            this->terminate_process();


            break;
        }


        // Update parameters
        this->update_params();
    }

    std::cout << "done." << std::endl;
    return;
}


int
Booster::
predict(const std::vector<double> &example)
{
    size_t T = this->classifiers.size();
    double confidence = 0.0;
    for (size_t t = 0; t < T; ++t)
        confidence += this->weights[t] * this->classifiers[t](example);
    int ans = (confidence > 0) ? 1 : -1;
    return ans;
}

std::vector<int>
Booster::
predict(const std::vector<std::vector<double> > &examples)
{
    const size_t M = examples.size();
    std::vector<int> ans(M);
    for (size_t m = 0; m < M; ++m)
        ans[m] = this->predict(examples[m]);
    return ans;
}


// ----------
