import csv, subprocess
parameter_file_full_path = "YOUR_PATH/file.csv"

with open(parameter_file_full_path,"rb") as csvfile:
        reader = csv.reader(csvfile)
        for job in reader:
		print job
            	qsub_command = """qsub -v n={0},p={1},kp={2},T={3},seed={4} julia_bic.pbs""".format(*job)
                exit_status=subprocess.call(qsub_command,shell=True)
            	if exit_status is 1:
                    print "Job {0} failed to submit.".format(qsub_command)
print "Done submitting jobs!"
