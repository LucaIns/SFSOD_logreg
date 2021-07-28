import csv, subprocess
parameter_file_full_path = "YOUR_PATH/file.csv"

with open(parameter_file_full_path,"rb") as csvfile:
        reader = csv.reader(csvfile)
        for job in reader:
		print job
            	qsub_command = """qsub -v kp={0},kn={1},maxtime={2},seed={3}, julia.pbs""".format(*job)
            	exit_status=subprocess.call(qsub_command,shell=True)
            	if exit_status is 1:
                    print "Job {0} failed to submit.".format(qsub_command)
print "Done submitting jobs!"



