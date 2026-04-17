import os
import shutil
import argparse

def organize_data(args):
    source_data_path = args.source
    meta_file_path = args.meta
    dest_dir = args.dest

    if source_data_path is None or meta_file_path is None or dest_dir is None:
        print('Please add paths to args')
        return

    skill_map = {
        'N': 'Novice',
        'E': 'Expert',
        'I': 'Intermediate'
    }

    stats = {'Novice': 0, 'Expert': 0, 'Intermediate': 0, 'Missing': 0}

    for skill in skill_map.values():
        dir_path = os.path.join(dest_dir, skill)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created folder: {dir_path}")

    print(f'Reading metadata from: {meta_file_path}')

    try:
        with open(meta_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print('Error: meta file not found')
        return
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        filename = parts[0]
        skill_char = parts[1]

        if skill_char not in skill_map:
            print(f"Skipping unknown skill code '{skill_char}' for file {filename}")
            continue
        target_skill = skill_map[skill_char]
        src_file = os.path.join(source_data_path, filename + '.txt')
        dest_path = os.path.join(dest_dir, target_skill)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)  
            print(f"Created folder: {dest_path}")
        dest_file = os.path.join(dest_dir, target_skill, filename + '.txt')

        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
            stats[target_skill] += 1
        else:
            print(f"Warning: Data file not found: {src_file}")
            stats['Missing'] += 1

    print("-" * 30)
    print("Organization Complete!")
    print(f"Novice Files:       {stats['Novice']}")
    print(f"Expert Files:       {stats['Expert']}")
    print(f"Intermediate Files: {stats['Intermediate']}")
    print(f"Missing Files:      {stats['Missing']}")
    print(f"Data is now in:     {os.path.abspath(dest_dir)}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        default=None,
        type=str
    )
    parser.add_argument(
        '--meta',
        default=None,
        type=str
    )
    parser.add_argument(
        '--dest',
        default=None,
        type=str
    )
    args = parser.parse_args()
    organize_data(args)