# path
user_path = cd("YOUR_PATH")
# path for R packages
user_pathR = string("YOUR_PATH")

# import Pkg
using JuMP
using RCall
using Gurobi
using DataFrames
using CSV
import MathOptInterface
const MOI = MathOptInterface
using MosekTools
using Random

kp = ARGS[1];
kp = parse(Int,kp);
kn = ARGS[2];
kn = parse(Int,kn);
maxtime = ARGS[3];
maxtime = parse(Int,maxtime);
seed = ARGS[4];
seed = parse(Int,seed);

Random.seed!(seed);

#-------------------------------------------------------------
# Function to generate date for simulations. Parameters:
# n - sample size
# p - no. features
# true_beta - regression coefficient
# kp - no. relevant features
# intercept - true/false include/exclude intercept
# seed - random seed for data generation
#-------------------------------------------------------------
function generate_dataset(n=100, p=10; true_beta=5, kp=p, kn=0, intercept = true)
    
    X = randn(n, p)
    if intercept == true
        X = hcat(ones(n, 1), X)
    end

    # sets of irrelevant (first p-kp) and relevant features (last kp)   
    irrp = (1+intercept):(intercept+p-kp);
    relp = (1+intercept+p-kp):(p+intercept); # excluding intercept (but counted too)

    w = ones(p+intercept, 1) .* true_beta
    w[irrp] .= 0;
    z = X * w;
    pr = 1 ./ (1 .+ exp.(-z)) 
    @rput n
    @rput pr
    R"""
        set.seed(1)
        # set.seed(10)
        y = rbinom(n, 1, pr) 
        y[y==0] = -1
    """
    y = rcopy(R"y");

    if kn>0
        # add contamination (relevant predictors and switching y labelling)
        X[1:kn, relp] = abs.(X[1:kn, relp]) .+ 10;
        y[1:kn] .= -1;
    end

    # println(X)
    # println(y)
    # println(pr)

    return X, y
end


#-------------------------------------------------------------
# Function to add conic constraints to MIP. Parameters:
# model - current MIP model to add constraints to
# t - t variables from MIP model
#     corresponds to deviances
# u - u variabes from MIP model
#     corresponds to fitted values (X^TB + phi)x y
#-------------------------------------------------------------
function exp_conic(model, t, u)
    z = @variable(model, [1:2], lower_bound = 0.0)
    @constraint(model, sum(z) <= 1.0)
    @constraint(model, [u - t, 1, z[1]] in MOI.ExponentialCone())
    @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
end


function sfsod_logreg(X, y, λ, kp, kn, maxtime)
    n, p = size(X)

    # initialize model (using Mosek)
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_OPTIMIZER_MAX_TIME"=>maxtime,
        "MSK_DPAR_MIO_TOL_REL_GAP"=>0.025))

    # optimization variables
    @variable(model, β[1:p])      # coefficients for features
    @variable(model, ϕ[1:n])      # coefficients for observations
    @variable(model, t[1:n])      # deviances
    @variable(model, Z[1:p],Bin)  # Binary auxillary variables for beta
    @variable(model, Zp[1:n],Bin) # Binary auxillary variables for phi

    # objective, minimize deviances + regularization (optional)
    @objective(model, Min, sum(t) + λ * reg)

    # big-M bounds for beta
    M = ones(p, 1).*3;
    @constraint(model, cons2[i = 1:p], -(M[i]*Z[i]) <= θ[i]);
    @constraint(model, cons3[i = 1:p], (M[i]*Z[i]) >= θ[i]);
    
    # force intercept in
    @constraint(model, cons4, Z[1] == 1);
    
    # big-M bounds for phi
    Mp = ones(n, 1).*80;
    @constraint(model, cons6[i = 1:n], -(Mp[i]*Zp[i]) <= ϕ[i]);
    @constraint(model, cons7[i = 1:n], (Mp[i]*Zp[i]) >= ϕ[i]);

    # sparsity constraint for beta
    @constraint(model, cons1, sum(Z[i] for i in 1:p) <= kp);

    # trimming constraint for phi    
    @constraint(model, cons5, sum(Zp[i] for i in 1:n) <= kn);

    # compute fited values for each observation and add conic constraints
    for i in 1:n
        u = - (X[i, :]' * θ + ϕ[i]) * y[i]
        exp_conic(model, t[i], u)
    end

    # Add ℓ2 regularization constraint (optional)
    @variable(model, 0.0 <= reg)
    @constraint(model, [reg; θ] in MOI.SecondOrderCone(p+1))
    
    ####################
    #  honey bee data
    ####################
    # group constraint for dummy variables
    @variable(model, G[1],Bin);
    @constraint(model, Z[20:23] .== G[1]);

    return model
end


# Load data
@rput user_path;
@rput seed;
R"""
    setwd(user_path)
    set.seed(seed)
    load("beePAtrain.RData")
    dat = data
    
	# 1 and -1 labeling
    y = dat$survival
    n = length(y)
    y[y==0] = -1
    X = as.matrix(dat[,2:ncol(dat)])
    X = cbind(rep(1, n), X) 

    # 0-1 labeling with noise on dummies
    y0 = dat$survival
    X0 = dat[,2:ncol(dat)]
    require(MASS)
	    ptmp = 4
	    tmperr = mvrnorm(n, rep(0, ptmp), diag(ptmp)*0.0001)
	    X0[, (ncol(X0)-3):ncol(X0)] = X0[, (ncol(X0)-3):ncol(X0)] + tmperr

"""
intercept = true;
y           = rcopy(R"y");
X           = rcopy(R"X");
y0          = rcopy(R"y0");
X0          = rcopy(R"X0");

est_nam = ["enetLTS","MIProb","MIP"]; # ,"MILP"];
n_est = length(est_nam);
solb     =   zeros(size(X)[2], n_est);
solw     =   zeros(length(y), n_est);
solr     =   zeros(length(y), n_est);

# path
R_lib_path = string(user_pathR, "Rpack", "")
@rput R_lib_path
R"""
    .libPaths(c(.libPaths(), R_lib_path))
""" 
        
@rlibrary enetLTS
trimlev = kn/length(y);
@rput trimlev;
R"""
      options(warn=-1)
      enetLTS_sol <- enetLTS:::enetLTS(X0, y0, family="binomial", alphas = 1, 
                                        hsize = 1-trimlev, nsamp=1000, intercept=1,
                                        plot = FALSE, nfold = 5, para=TRUE, ncores=24)
""";
      
est_i = 1;
solb[:, est_i] = rcopy(R"c(enetLTS_sol$a00, enetLTS_sol$raw.coefficients)");
solw[:, est_i] = rcopy(R"enetLTS_sol$raw.wt");
solr[:, est_i] = rcopy(R"enetLTS_sol$raw.residuals");
est_i += 1;

# MIProb: solve the problem (robustly)
λ = 0
model = sfsod_logreg(X, y, λ, kp+intercept, kn, maxtime)
JuMP.optimize!(model)
solb[:, est_i] = JuMP.value.(model[:θ]);
tmpw = JuMP.value.(model[:Zp]);
ind0 = tmpw .== 0;
ind1 = tmpw .== 1;
solw[ind0, est_i] .= 1;
solw[ind1, est_i] .= 0;
solr[:, est_i] = JuMP.value.(model[:ϕ]);
est_i += 1;

# MIP: solve the problem (non-robustly)
model = sfsod_logreg(X, y, λ, kp+intercept, 0, maxtime)
JuMP.optimize!(model)
solb[:, est_i] = JuMP.value.(model[:θ]);
solw[:, est_i] .= 1;
solr[:, est_i] = JuMP.value.(model[:ϕ]);
est_i += 1;

# save output
solb = DataFrame(solb);
solw = DataFrame(solw);
solr = DataFrame(solr);
rename!(solb, est_nam);
rename!(solw, est_nam);
rename!(solr, est_nam);
cd(string(user_path, "/output"));
tit = join((kp,kn,maxtime,),"-")
CSV.write(join((tit, "solb.csv"), "-"), solb);
CSV.write(join((tit, "solw.csv"), "-"), solw);
CSV.write(join((tit, "solr.csv"), "-"), solr);