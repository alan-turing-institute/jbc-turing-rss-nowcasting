"""
This model is the same as in Jersakova et al. (2021) except a hyperprior is 
introduced for the random walk length and inference is done using NUTS instead 
of SMC.

Weekend effects not yet included.

The original model includes the discrete latent variables X, which 
represent the number of tests from a particular day that will return positive.
In order to use HMC, these are marginalised out. They'll still be returned after 
running stan, as they are produced in the generated quantities block.
"""
data {
  // measured positive tests over a sequence of days
  int<lower=1> n_days;
  int y[n_days];

  // empirical bayes parameters for priors on fraction of cases reported
  vector[n_days] a;
  vector[n_days] b;
}
transformed data {
   // gamma parameters for the initial poisson rate
   real p = 0.3 * pow(y[1], 2);
   real q = 0.3 + y[1];
}
parameters {

  // temporal smoothing (section 5.3)
  vector[n_days - 2] epsilon;
  real<lower=0> sigma_rw;
  real kappa_1;
  real<lower=0> lambda_1;

  // binomial parameters
  vector<lower=0, upper=1>[n_days] theta;
}
transformed parameters {
   vector<lower=0>[n_days] lambda;
   vector[n_days - 1] kappa;
   lambda[1] = lambda_1;
   kappa[1] = kappa_1;

   {  
    for (t in 2:n_days - 1) {
      lambda[t] = kappa[t - 1] + lambda[t - 1];
      kappa[t] = kappa[t - 1] + sigma_rw * epsilon[t - 1];
    }

    lambda[n_days] = kappa[n_days - 1] + lambda[n_days - 1];
   }    

} 
model {
  epsilon ~ std_normal();
  sigma_rw ~ normal(0, 5);
  kappa_1 ~ normal(0, sigma_rw);
  lambda_1 ~ gamma(p, q);
  theta ~ beta(a, b);

  // likelihood with latent true number of positive tests marginalised out 
  y ~ poisson(theta .* lambda);
}
generated quantities {
  // produce samples of the latent true number of positive tests
  int[n_days] x;

  for (t in 1:n_days) {
    x[t] = poisson_rng(theta[t] * lambda[t]);
  }

}
