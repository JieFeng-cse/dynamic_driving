import os
import sys
import shutil

for root, dir_list, file_list in os.walk('/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles'):
    for dirr in dir_list:
        if dirr not in ['segmented','sorted','train','val','.TestScenarioForScripts']:
            print(os.path.join(root,dirr,'train','sorted'))
            if os.path.exists(os.path.join(root,dirr,'train','sorted')):
                shutil.rmtree(os.path.join(root,dirr,'train','sorted'))
                shutil.rmtree(os.path.join(root,dirr,'train','segmented'))
                shutil.rmtree(os.path.join(root,dirr,'val','sorted'))
                shutil.rmtree(os.path.join(root,dirr,'val','segmented'))

