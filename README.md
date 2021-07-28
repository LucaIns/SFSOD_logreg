# MIProb - SFSOD for logistic regression

Methods for “Robust Variable Selection with Optimality Guarantees for High-Dimensional Logistic Regression” (by L. Insolia, A. Kenney, M. Calovi and F. Chiaromonte)

This project contains the source code to replicate our analyses.
Specifically, the application folder contains:

	- julia.pbs: PBS file to submit jobs over different simulation parameters on cluster

	- sumbit.py: python script to submit all jobs based on PBS file, loops through parameter settings through file.csv

	- file.csv: csv file listing settings presented in main manuscript

	- sfsod_logreg.jl: generating data and running MIProb/MIP

	- results: contains our results, as well as the code to reproduce our Figures and Tables

	- output: default path to save output results (betas, phi, etc.)

	- LOG: default path to save log files for various jobs

The simulations folder contains:

	- sfsod_logreg_bic.jl 
		- main julia script for generating data and running MIProb/MIP
		- update user_path_finalsol and user_path_fullsol for directories to store final solutions (after tuning) and all solutions under all sparsity bounds respectively

	- julia_bic.pbs: PBS file to submit jobs over different simulation parameters on cluster
	
	- submit.py: python script to submit all jobs based on PBS file, loops through parameter settings through file.csv
	
	- file.csv: csv file listing settings presented in main manuscript
	
	- PerformanceMeasures.jl
		- julia script to loop through iterations and compute main performance measures considered in manuscript to generate final table
		- update directories (see simulation_results.zip):
			- user_path_finalsol directory with final solutions
			- user_path_fullsol directory with full set of solutions and computing time
			- user_path_finalsol_oracle directory storing solutions under oracle (if different from user_path_finalsol)
			- user_path_finalsol_enet directory storing solutions under enetLTS (if different from user_path_finalsol)
			- user_path_summary directory to store final summary file

	- ComputingTimePlot.R
		- R script to plot Figure 1 in main manuscript
		- update user_path_computingtime for directory containing relevant computing time results
	
To replicate our analyses the user should only change the file path (YOUR_PATH variable) at the beginning of .pbs, .py and .jl scripts. 

Our code is currently implemented in Julia (v.1.3.1) interacting with Mosek through its JuMP package, and R (v.3.5.2).

The honey bee data in use are publicly available (see Section 5.1).

For any problems or comments feel free to contact the correponding author (Luca.insolia@sns.it)