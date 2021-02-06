## pipeline001.py -- prediction for Things, with maxpool

import os
if __name__ == '__main__':
    #rois = ['EVC', 'LOC', 'PFS', 'OPA', 'PPA', 'RSC', 'FFA', 'OFA', 'STS', 'EBA']
    rois = ['LOC']
    model = ['alexnet']
    layers = ['5']
    #layers = ['1', '2', '3', '4', '5', '6', '7']
    subjects = ['1', '2', '3', '4']
    for net in model:
        for subj in subjects:
            for roi in rois:
                for layer in layers:
                    print(subj, net, layer, roi)
                    name = str(subj) + "_" + net + layer + "_" + roi + "_maxpool"
                    cmd = "python predict.py --name " + name + " --stimuli_folder things" #e.g. 1_alexnet5_LOC_maxpool

                    print(cmd)
                    os.system(cmd)

                    #TBD semantic dimensionality plot
