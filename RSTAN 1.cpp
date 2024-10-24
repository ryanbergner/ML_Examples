write("// 
data {
int<lower=0> N; // Number of rows
int<lower=0> y [N]; // Counts
int<lower=0> n[N] // Binomial Totals
vector[N] x; // covariatepredictor
}

parameters {
real beta;
real betal;
}
               
transformed parameters {
vector [N] logit_p;
// Linear predictor
logit_p = beta0 + betal*x;
}
               
model {
// Priors
hetad ~ uniform(-5, 10).

parameters {
real beta0;
real betal;
}
    
transformed parameters {
vector [N] logit_p;
// Linear predictor
logit_p = betal + beta1*x;
}
    
model {
// Priors
beta ~ uniform(-5, 10);
betal ~ uniform(-10, 40);
// Likelohood
y ~ binomial(n, inv_logit(logit_p));

generated quantities {
real<lower=0, upper=1> b [NI;
//int<lower=0> y_rep [N];
for (i in I:N){
    p[il = inv_logit(logit_p[i]);
    //y_replil = binomial(n[i],p[il);
}
}",