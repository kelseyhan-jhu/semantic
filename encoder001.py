## encoder001.py -- vanila encoder for object2vec

import os
if __name__ == '__main__':
    rois = ['EVC', 'LOC', 'PFS', 'OPA', 'PPA', 'RSC', 'FFA', 'OFA', 'STS', 'EBA']
    model = ['alexnet']
    layers = ['1', '2', '3', '4', '5', '6', '7']
    subjects = ['1', '2', '3', '4']
    for subj in subjects:
            for roi in rois:
                    for layer in layers:
                            cmd = "python train.py --roi " + roi + " --layer " + layer + " --subject_number " + subj
                            print(cmd)
                            os.system(cmd)

# ROIs = ['LOC', 'PPA', 'EVC']
# layers = ['conv1', 'conv5', 'fc6']
# subjects = ['subj002', 'subj003', 'subj004']
# for subj in subjects:
#         for roi in ROIs:
#                 for layer in layers:
#                         cmd = "python predict.py --name " + roi + "_" + layer + "_" + subj + "_r" + " --stimuli_folder \"images\" --save_folder predicted/" + subj + "/" + roi + "_" + layer + "_r"
#                         print(cmd)
#                         os.system(cmd)