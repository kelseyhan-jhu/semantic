## encoder002.py -- encoder for object2vec with global maxpooling on convolutioal features

import os
if __name__ == '__main__':
    rois = ['EVC', 'LOC', 'PFS', 'OPA', 'PPA', 'RSC', 'FFA', 'OFA', 'STS', 'EBA']
    model = ['alexnet']
    layers = ['1', '2', '3', '4', '5', '6', '7']
    subjects = ['1', '2', '3', '4']
    for subj in subjects:
            for roi in rois:
                    for layer in layers:
                            cmd = "python train.py --roi " + roi + " --layer " + layer + " --subject_number " + subj + " --maxpool maxpool"
                            print(cmd)
                            os.system(cmd)