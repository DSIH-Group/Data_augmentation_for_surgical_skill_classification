import glob
import os
import shutil

dataset = 'data/jigsaws-data/'
surgery = 'Suturing'
meta_file = glob.glob(dataset+surgery+'/'+'meta*.txt')
skills = ['N', 'I', 'E']
files = {}

with open(meta_file[0], 'r') as f:
    for skill in skills:
        files[skill] = []
        for lines in f:
            line = lines.strip().split('\t')
            if len(line) == 0:
                continue
            if line[2] == skill:
                files[skill].append(line[0])
        f.seek(0)


def copy_files():
    for skill in skills:
        for file in files[skill]:
            skill_ = ''
            if skill == 'N':
                skill_ = 'Novice'
            elif skill == 'I':
                skill_ = 'Intermediate'
            else :
                skill_ = 'Expert'
            dest_dir = dataset + surgery + '/splits/' + skill_
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(dataset+surgery+'/kinematics/AllGestures/'+file+'.txt', dataset+surgery+'/splits/'+skill_)

def split_surgeons():
    for skill in skills:
        if skill == 'N':
            skill = 'Novice'
        elif skill == 'I':
            skill = 'Intermediate'
        else :
            skill = 'Expert'
        skill_files = glob.glob(dataset+surgery+'/splits/'+skill+'/*.txt')
        print(skill_files)
        for file in skill_files:
            file_ = file.strip().split('_')
            os.makedirs(dataset+surgery+'/splits/'+skill+'/'+file_[-1][0], exist_ok=True)
            shutil.move(file, dataset+surgery+'/splits/'+skill+'/'+file_[-1][0])

split_surgeons()