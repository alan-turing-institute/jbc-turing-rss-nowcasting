/*
This model is the same as in Jersakova et al. (2021) except a hyperprior is 
introduced for the random walk length and inference is done using NUTS instead 
of SMC.
*/
data {
  // measured positive tests over a sequence of days
  int<lower=1> n_days;
  int y[n_days];

  int is_weekend[n_days];

  // empirical bayes parameters for priors on fraction of cases reported
  vector[n_days] a;
  vector[n_days] b;
}
transformed data {
   // gamma parameters for the initial poisson rate
   real p = 0.3 * pow(y[1], 2);
   real q = 0.3 * y[1];

   int n_weekend_days = sum(is_weekend);
}
parameters {

  // temporal smoothing (section 5.3)
  vector[n_days - 2] epsilon;
  real<lower=0> sigma_rw;
  real kappa_1;
  real<lower=0> lambda_1;

  // weekend effects
  vector<lower=0, upper=1>[n_weekend_days] z_sparse;

  // binomial parameters
  vector<lower=0, upper=1>[n_days] theta;
}
transformed parameters {
   vector<lower=0>[n_days] lambda;
   vector[n_days - 1] kappa;
   vector[n_days] z = rep_vector(1.0, n_days);

  // fill the weekend effect vector
  {
    int weekend_index = 1;
    for (t in 1:n_days) {
      if (is_weekend[t] == 1) {
        z[t] = z_sparse[weekend_index];
        weekend_index += 1;
      }
    }
  }

  // fill the latent rate and local trend vectors
  lambda[1] = lambda_1;
  kappa[1] = kappa_1;
  for (t in 2:n_days - 1) {
    lambda[t] = kappa[t - 1] + lambda[t - 1];
    kappa[t] = kappa[t - 1] + sigma_rw * epsilon[t - 1];
  }
  lambda[n_days] = kappa[n_days - 1] + lambda[n_days - 1];

  }    
model {
  epsilon ~ std_normal();
  sigma_rw ~ normal(0, 5);
  kappa_1 ~ normal(0, sigma_rw);
  lambda_1 ~ gamma(p, q);
  theta ~ beta(a, b);
  z_sparse ~ beta(1, 1);

  // likelihood with latent true number of positive tests marginalised out 
  y ~ poisson(theta .* lambda .* z);
}
