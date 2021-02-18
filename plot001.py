## plot001.py -- produce plots for alexnet all layers and rois from pipeline001.txt and pipeline001_random.txt

import os
if __name__ == '__main__':
    rois = ['EVC', 'LOC', 'PFS', 'OPA', 'PPA', 'RSC', 'FFA', 'OFA', 'STS', 'EBA']
    model = ['alexnet']
    layers = ['2', '3', '4', '5', '6', '7']
    
    for layer in layers:
        keywords = []
        for roi in rois:
            keywords.append('alexnet' + layer + '_' + roi)
    
        cmd = "python plot.py --files pipeline001.txt pipeline001_random.txt --keywords " + ' '.join(keywords)
        print(cmd)
        os.system(cmd)
