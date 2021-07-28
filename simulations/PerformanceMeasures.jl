# ############################################################
# # julia calling Mosek through JuMP
# ############################################################

# path for R packages
user_path_finalsol = string("YOUR_PATH/FinalResults")
user_path_fullsol = string("YOUR_PATH/FullResults")
user_path_summary = string("YOUR_PATH/Summary")
user_path_finalsol_oracle = string("YOUR_PATH/OracleResults")
user_path_finalsol_enet = string("YOUR_PATH/EnetLTSResults")
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
using Statistics

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

#-------------------------------------------------------
# parameters
#-------------------------------------------------------
n, p, kp, maxtime = 100, 20, 3, 2400;
true_beta = 3;
kn = 5;
intercept = true;
tuning = "BIC";
@rput p;
@rput kp;
@rput kn;
@rput n;
cd(user_path_finalsol)
R"""
	library(stringr)
	files = Sys.glob(paste(n,p,kp+1,kn,"*2400-BIC-solw.csv",sep="-"))
	idx = vector(,length(files))
	for(i in 1:length(files)){
		all = str_split(files[i],"-")[[1]]
		idx[i] = as.numeric(all[5])
	}
"""
files = rcopy(R"files");
idx = Int.(rcopy(R"idx"));
rep_tot = size(idx)[1];
n_est = 5;
irrp = (1+intercept):(intercept+p-kp);
relp = (1+intercept+p-kp):(p+intercept); # excluding intercept (but counted too)
beta_true = ones(p+intercept, 1) .* true_beta;
beta_true[irrp] .= 0;

#-------------------------------------------------------
# load in elastic net and oracle solutions
#-------------------------------------------------------
cd(user_path_finalsol_enet)
opt_str = join((n, p, kp+1, kn), "-");
B_enet = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"solb.csv"),"-"))));
Phi_enet = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"solw.csv"),"-"))));
COMP_enet = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"COMP.csv"),"-"))));

cd(user_path_finalsol_oracle)
opt_str = join((n, p, kp+1, kn), "-");
B_oracle = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"solb.csv"),"-"))));
Phi_oracle = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"solw.csv"),"-"))));
COMP_oracle = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"COMP.csv"),"-"))));


sol = zeros(rep_tot, 9, n_est);
B_est = zeros(rep_tot,p+1,n_est);
Phi_est = zeros(rep_tot,n,n_est) .+ 1;
seedcnt = 0;
for seed in idx
	global seedcnt += 1 
	Xtest, Ytest = generate_dataset(n, p, true_beta=true_beta, kp=kp, kn=0, intercept = intercept, seed = seed*100);
	println("generate test data")
	#-------------------------------------------------------
	# oracle
	#-------------------------------------------------------
	B_est[seedcnt,:,1] = B_oracle[:,seed];
	Phi_est[seedcnt,1:kn,1] .= 0;
	sol[seedcnt,9,1] = COMP_oracle[seed];
	#-------------------------------------------------------
	# enet
	#-------------------------------------------------------
	B_est[seedcnt,:,2] = B_enet[:,seed];
	Phi_est[seedcnt,:,2] = Phi_enet[:,seed];
	sol[seedcnt,9,2] = COMP_enet[seed];
	#-------------------------------------------------------
	# read in results
	#-------------------------------------------------------
	opt_str = join((n, p, kp+1, kn, seed, maxtime,tuning), "-");
	println(opt_str)
	cd(user_path_finalsol)
	B_est[seedcnt,:,3:4] = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"solb.csv"),"-"))))[:,2:3];
	Phi_est[seedcnt,:,3:4] = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"solw.csv"),"-"))))[:,2:3];
	cd(user_path_fullsol)
	tmp = convert(Matrix{Float64},DataFrame(CSV.File(join((opt_str,"COMP.csv"),"-"))))[:,2:3];
	sol[seedcnt,9,3:4] = [sum(tmp[:,1]), sum(tmp[:,2])]
	#-------------------------------------------------------
	# fit lasso
	#-------------------------------------------------------
	X, y = generate_dataset(n, p, true_beta=true_beta, kp=kp, kn=kn, intercept = intercept, seed = seed);
	y0 = y
    ind0 = y0.==-1
    y0[ind0] .= 0
    X0 = X[: ,2:size(X, 2)]
    @rput y0
    @rput X0
    start = time()
    @rput seed
	R"""
		RNGkind(sample.kind = "Rounding")
        set.seed(seed)
		library(glmnet)
		cv.lasso = cv.glmnet(x=X0,y=y0,nfolds=10,alpha=1,family="binomial")
		B_lasso = as.numeric(coef(cv.lasso,s=cv.lasso$lambda.min))
	"""
	sol[seedcnt,9,5] = time() - start
	B_est[seedcnt,:,5] = rcopy(R"B_lasso");
	for est_j = 1:n_est

		# MNLL
		pred = Xtest*B_est[seedcnt,:,est_j];
		sol[seedcnt,1,est_j] = mean(log.(1 .+ exp.(-Ytest.*pred)))
		prob = 1 ./ (1 .+ exp.(-pred)) 
		y0 = prob
		ind0 = y0.<0.50
		pred[ind0] .= -1;
		ind0 = y0.>=0.50
		pred[ind0] .= 1;

		# classification error
		res = Ytest - pred;
		correct = res.==0;
		sol[seedcnt, 2, est_j] = (n-sum(correct))/n;
		# FPR and FNR for beta
		fp = sum(B_est[seedcnt,irrp, est_j] .!= 0)/(p-kp);
		fn = sum(B_est[seedcnt,relp, est_j] .== 0)/(kp+1);
		sol[seedcnt, 5:6, est_j] = hcat(fp, fn);		
		# FPR and FNR for phi
		fp = sum(Phi_est[seedcnt,(1+kn):n, est_j] .!= 1)/(n-kn);
		if kn == 0 
			fn = sum(Phi_est[seedcnt,1:kn, est_j] .== 1)/(n);
		else
			fn = sum(Phi_est[seedcnt,1:kn, est_j] .== 1)/(kn);
		end
		sol[seedcnt, 7:8, est_j] = [fp, fn];
	end
end

# compute MSE decomposition
for est_j = 1:n_est
	B_j = B_est[:, :, est_j];
	BinitStar = mean(B_j, dims=1);
	BinitVar = mean(mean((B_j .- BinitStar).^2, dims=1));
	BinitBias2 = mean((BinitStar .- beta_true').^2);
	sol[:, 3:4, est_j] = hcat(BinitVar, BinitBias2) .* ones(rep_tot, 1);
end

est_nam = ["oracle", "enetLTS", "MIProb","MIP","Lasso"]
sol = reshape(sol, rep_tot, 9*n_est);
varNam = ["MNLL","ClassErr", "var", "bias", "fpBeta", "fnBeta", "fpPhi", "fnPhi", "time"];
colnames = []
for i = 1:length(est_nam)
	println(i)
    append!(colnames, est_nam[i] .* "_" .* varNam);
end
colnames = string.(colnames)
sol = DataFrame(sol);
rename!(sol, colnames);

opt_str = join((n, p, kp+1, kn, maxtime,tuning), "-");
cd(user_path_summary)
sav_nam = join((opt_str,"allfinal.csv"),"-");
CSV.write(sav_nam, sol);