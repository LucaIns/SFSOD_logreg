# ############################################################
# # julia calling Mosek through JuMP
# ############################################################

# path result storage
user_path_finalsol = string("YOUR_PATH/FinalResults")
user_path_fullsol = string("YOUR_PATH/FullResults")

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

#-------------------------------------------------------------
# Function to generate date for simulations. Parameters:
# n - sample size
# p - no. features
# true_beta - regression coefficient
# kp - no. relevant features
# intercept - true/false include/exclude intercept
# seed - random seed for data generation
#-------------------------------------------------------------
function generate_dataset(n=100, p=10; true_beta=5, kp=p, kn=0, intercept = true,seed=1)
    Random.seed!(seed);
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
        # z[1:kn] .= 0
    pr = 1 ./ (1 .+ exp.(-z)) 
    @rput n
    @rput pr
    @rput seed
    R"""
        RNGkind(sample.kind = "Rounding")
        set.seed(seed)
        y = rbinom(n, 1, pr) 
        y[y==0] = -1
    """
    y = rcopy(R"y");

    if kn>0
        # add contamination (relevant predictors and switching y labelling)
        X[1:kn, relp] = abs.(X[1:kn, relp]) .+ 10;
        y[1:kn] .= -1;
    end

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

#-------------------------------------------------------------
# Function to generate main MIP model. Parameters:
# X - design matrix (include column of 1's for intercept)
#     remeber to standardize if regularizing
# y - response vector
# λ - optional ridge regularization parameter, set to 0 if none
# kp - sparsity constraint on features (includes intercept)
# kn - triming constraint for observations
# maxtime - max time set before ending algorithm
# Cb - constant for M bounds of coefficients for features (beta)
#      keep wide enough for accuracy
# Cp - constant for M bounds of coefficeints for observations (phi)
# threads - no. of threads when parallelizing
#-------------------------------------------------------------
function MIP_logreg(X, y, λ, kp, kn, maxtime,Cb,Cp, threads)
    n, p = size(X)

    # initialize model (using Mosek)
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_OPTIMIZER_MAX_TIME"=>maxtime,
        "MSK_DPAR_MIO_TOL_REL_GAP"=>0.025,"MSK_IPAR_NUM_THREADS"=>threads))

    # optimization variables
    @variable(model, β[1:p])      # coefficients for features
    @variable(model, ϕ[1:n])      # coefficients for observations
    @variable(model, t[1:n])      # deviances
    @variable(model, Z[1:p],Bin)  # Binary auxillary variables for beta
    @variable(model, Zp[1:n],Bin) # Binary auxillary variables for phi

    # objective, minimize deviances + regularization (optional)
    @objective(model, Min, sum(t) + λ * reg)


    # big-M bounds for beta
    M = ones(p, 1).*Cb;
    @constraint(model, cons2[i = 1:p], -(M[i]*Z[i]) <= β[i]);
    @constraint(model, cons3[i = 1:p], (M[i]*Z[i]) >= β[i]);

    # big-M bounds for phi
    Mp = ones(n, 1).*Cp
    @constraint(model, cons6[i = 1:n], -(Mp[i]*Zp[i]) <= ϕ[i]);
    @constraint(model, cons7[i = 1:n], (Mp[i]*Zp[i]) >= ϕ[i]);

    # force intercept in
    @constraint(model, cons4, Z[1] == 1);

    # sparsity constraint for beta
    @constraint(model, cons1, sum(Z[i] for i in 1:p) <= kp);

    # trimming constraint for phi
    @constraint(model, cons5, sum(Zp[i] for i in 1:n) <= kn);

    # compute fited values for each observation and add conic constraints
    for i in 1:n
        u = - (X[i, :]' * β + ϕ[i]) * y[i]
        exp_conic(model, t[i], u)
    end

    # Add ℓ2 regularization constraint (optional)
    @variable(model, 0.0 <= reg)
    @constraint(model, [reg; β] in MOI.SecondOrderCone(p+1))

    return model
end


#-------------------------------------------------------------
# example code for running a simulation scenario
#-------------------------------------------------------------

# read in simulation parameters
n = ARGS[1];      # sample size
n = parse(Int,n);

p = ARGS[2];      # no. features 
p = parse(Int,p);

kp = ARGS[3];     # no. relevant features
kp = parse(Int,kp);

maxtime = ARGS[4];  # max time for MIP
maxtime = parse(Int,maxtime);

seed = ARGS[5];    # random seed for replication
seed = parse(Int,seed);

intercept = true;  # whether to include intercept
maxk = minimum([kp + 5 p+1]);   # max sparsity bound in features to tune over
true_beta = 3;     # feature regression coefficients
kn = 5;            # no. of observations to trim
tuning = "BIC";    # tuning method
Cb = 10;           # M bound constant for features
Cp = 50;           # M bound constant for observations
threads = 24;      # no. of threads for parallelizing

opt_str = join((n, p, kp+1, kn, seed, maxtime,tuning), "-");
println(opt_str)


# generate data
X, y = generate_dataset(n, p, true_beta=true_beta, kp=kp, kn=kn, intercept = intercept, seed = seed);
println("generate data")

est_nam = ["enetLTS","MIProb","MIP"];
n_est = length(est_nam);
solb     =   zeros(size(X)[2], n_est);
solw     =   zeros(length(y), n_est);
solr     =   zeros(length(y), n_est);
solt     =   zeros(length(y), n_est);
COMP     = zeros(maxk, n_est);
# path
R_lib_path = string(user_path_finalsol, "Rpack", "")
@rput R_lib_path
R"""
    .libPaths(c(.libPaths(), R_lib_path))
""" 

# run enetLTS for comparison
@rlibrary enetLTS
trimlev = 0.2 # amount of trimming for enetLTS
@rput trimlev;
y0 = y
ind0 = y0.==-1
y0[ind0] .= 0
X0 = X[: ,2:size(X, 2)]
@rput y0
@rput X0
@rput threads
start = time()
R"""
      options(warn=-1)
      enetLTS_sol <- enetLTS:::enetLTS(X0, y0, family="binomial", alphas = 1,lambdas=0.05,
                                        hsize = 1-trimlev, nsamp=500, intercept=1,
                                        plot = FALSE, nfold = 5, para=TRUE, ncores=threads)
    # enetLTS_sol = enetLTS:::enetLTS(X0,y0,family="binomial",alphas=1,hsize=1-trimlev,lambdas=0.05,plot=FALSE)
""";

est_i = 1;
solb[:, est_i] = rcopy(R"c(enetLTS_sol$a00, enetLTS_sol$raw.coefficients)");
solw[:, est_i] = rcopy(R"enetLTS_sol$wt");
solr[:, est_i] = rcopy(R"enetLTS_sol$raw.residuals");
solt[:, est_i] .= -1;
COMP[:, est_i] .= time() - start
est_i += 1;

# run MIProb and compute BIC over sparsity bounds
λ = 0 # no regularization
BIC      = zeros(maxk, n_est-1);
solb_tmp = zeros(size(X)[2],maxk,n_est-1)
solw_tmp = zeros(length(y),maxk,n_est-1)
solr_tmp = zeros(length(y),maxk,n_est-1)
solt_tmp = zeros(length(y),maxk,n_est-1)
for k_s = 1:maxk
    println(k_s)
    model = MIP_logreg(X, y, λ, k_s, kn, maxtime,Cb,Cp, threads)
    JuMP.optimize!(model)
    COMP[k_s,2] = solve_time(model);
    solb_tmp[:,k_s,1] = JuMP.value.(model[:β]);
    solt_tmp[:, k_s,1] = JuMP.value.(model[:t]);
    tmpw = JuMP.value.(model[:Zp]);
    ind0 = tmpw .== 0;
    ind1 = tmpw .== 1;
    solw_tmp[ind0, k_s,1] .= 1;
    solw_tmp[ind1, k_s,1] .= 0;
    solr_tmp[:, k_s,1] = JuMP.value.(model[:ϕ]);
    BIC[k_s,1] = k_s*log(sum(solw_tmp[:,k_s,1])) + sum(solt_tmp[:,k_s,1]);
end
kopt = findmin(BIC[:,1])[2] # take k_s with minimum
solb[:,est_i] = solb_tmp[:,kopt,1];
solt[:,est_i] = solt_tmp[:,kopt,1];
solw[:,est_i] = solw_tmp[:,kopt,1];
solr[:,est_i] = solr_tmp[:,kopt,1];
est_i += 1;

# solve MIP (non-robust)
for k_s = 1:maxk
    model = MIP_logreg(X, y, λ, k_s, 0, maxtime)
    JuMP.optimize!(model)
    COMP[k_s,3] = solve_time(model);
    solb_tmp[:,k_s,2] = JuMP.value.(model[:β]);
    solt_tmp[:, k_s,2] = JuMP.value.(model[:t]);
    solw_tmp[:, k_s,2] .= 1;
    solr_tmp[:, k_s,2] = JuMP.value.(model[:ϕ]);
    BIC[k_s,2] = k_s*log(sum(solw_tmp[:,k_s,2])) + sum(solt_tmp[:,k_s,2]);
end
kopt = findmin(BIC[:,2])[2] # take k_s with minimum
solb[:,est_i] = solb_tmp[:,kopt,2];
solt[:,est_i] = solt_tmp[:,kopt,2];
solw[:,est_i] = solw_tmp[:,kopt,2];
solr[:,est_i] = solr_tmp[:,kopt,2];

 
# save output
solb = DataFrame(solb);
solt = DataFrame(solt);
solw = DataFrame(solw);
solr = DataFrame(solr);
solb_all_rob = DataFrame(solb_tmp[:,:,1]);
solb_all_nrob = DataFrame(solb_tmp[:,:,2]);
solr_all_rob = DataFrame(solr_tmp[:,:,1]);
solr_all_nrob = DataFrame(solr_tmp[:,:,2]);
solw_all_rob = DataFrame(solw_tmp[:,:,1]);
solw_all_nrob = DataFrame(solw_tmp[:,:,2]);
BIC = DataFrame(BIC);
COMP = DataFrame(COMP);
rename!(solb, est_nam);
rename!(solt, est_nam);
rename!(solw, est_nam);
rename!(solr, est_nam);
rename!(BIC, est_nam[2:3]);
rename!(COMP, est_nam);
allk = string.([1:1:maxk;])
rename!(solb_all_rob,allk);
rename!(solb_all_nrob,allk);
rename!(solr_all_rob,allk);
rename!(solr_all_nrob,allk);
rename!(solw_all_rob,allk);
rename!(solw_all_nrob,allk);
#-------------------------------------------------------------------------
# Store all Results
#-------------------------------------------------------------------------
cd(user_path_finalsol)
sav_nam = join((opt_str,"solb.csv"),"-");
CSV.write(sav_nam, solb);
sav_nam = join((opt_str,"solw.csv"),"-");
CSV.write(sav_nam, solw);
sav_nam = join((opt_str,"solr.csv"),"-");
CSV.write(sav_nam, solr);

cd(user_path_fullsol)
sav_nam = join((opt_str,"BIC.csv"),"-");
CSV.write(sav_nam, BIC);
sav_nam = join((opt_str,"COMP.csv"),"-");
CSV.write(sav_nam, COMP);
sav_nam = join((opt_str,"solb_all_rob.csv"),"-");
CSV.write(sav_nam, solb_all_rob);
sav_nam = join((opt_str,"solb_all_nrob.csv"),"-");
CSV.write(sav_nam, solb_all_nrob);
sav_nam = join((opt_str,"solw_all_rob.csv"),"-");
CSV.write(sav_nam, solw_all_rob);
sav_nam = join((opt_str,"solw_all_nrob.csv"),"-");
CSV.write(sav_nam, solw_all_nrob);
sav_nam = join((opt_str,"solr_all_rob.csv"),"-");
CSV.write(sav_nam, solr_all_rob);
sav_nam = join((opt_str,"solr_all_nrob.csv"),"-");
CSV.write(sav_nam, solr_all_nrob);