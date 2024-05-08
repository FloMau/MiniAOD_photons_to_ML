
# before running:
	- change directories in .sh files to the directories where you keep the python files
	- change (currently still hardcoded) python output paths in .py files 

- for grid access set proxy before submitting jobs with: 
	`voms-proxy-init --voms cms`

# what is what
- reminder.txt contains short info on activation of cmsenv, grid-proxy and tensorflow
- datafiles.txt contains all filenames of the dataset
	-create with: `dasgo-client -query=files _yourdataset_.root > datafiles.txt`

- to preprocess: `condor_submit preprocessing.sub`
	(currently works only on lxblade machines)
	- this will submit one job for each filename in datafiles.txt
	  the jobs will be numbered (three digits with leading zeros)
	- job outputs, logs, and error will be saved in outs/ errs/ logs/

- after completion (all.pkl, preselection.npy, rechits.npy are placeholders):
```
	python3 combine_files.py _your_dir_with_output_/*.pkl --outname all.pkl
	python3 get_preselection.py all.pkl preselection.npy
	python3 combine_files.py _your_dir_with_output_/*.npy --outname rechits.npy --preselection preselection.pkl
```
	- the last line saves only rechits of events passing preselection (otherwise the file will be too large)


