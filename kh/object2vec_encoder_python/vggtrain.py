import os
ROIs = ['LOC', 'PPA', 'EVC']
subjects = ['subj001', 'subj002', 'subj003', 'subj004']
for i, subj in enumerate(subjects):
	for roi in ROIs:
		cmd = "python train.py --name vgg\/" + roi + "_conv5_" + subj + " --rois " + roi + " --subject_number " + str(i+1) + " --resolution 256 --feature_extractor vgg16 --l2 0;"
		print(cmd)
		os.system(cmd)
