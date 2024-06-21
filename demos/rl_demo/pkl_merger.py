import os
import re
import numpy as np

def merge_from_folder(folder):
    episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl') and 'combined' not in f]
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]

    filenums = [re.findall('\d+',f) for f in filenames_only]
    final_filenums = []
    for i in filenums:
        if len(i) > 0 :
            final_filenums.append(int(i[0]))

    sorted_inds = np.argsort(final_filenums)
    final_filenums = np.array(final_filenums)
    episode_files = np.array(episode_files)
    filenames_only = np.array(filenames_only)
    episode_files = episode_files[sorted_inds].tolist()
    # print(episode_files[-1])

    import pickle as pkl
    import pandas as pd
    all_things = []
    for file in episode_files:
        with open(file,'rb') as fi:
            temp = pkl.load(fi)
        # print(temp)
        tduct = {'Start X':temp['Start Pos'][0],'Start Y':temp['Start Pos'][1]-0.1,
                'End X':temp['End Pos'][0],'End Y':temp['End Pos'][1]-0.1,
                'Goal X':temp['Goal Position'][0],'Goal Y':temp['Goal Position'][1],
                'End Distance':temp['End Distance'], 'Max Distance': temp['Max Distance'],
                'End Orientation':temp['End Orientation'],'Goal Orientation': temp['Goal Orientation'],
                'Path':''}
        all_things.append(tduct)

    full_df = pd.DataFrame(all_things)
    full_df['Rounded Start X'] = full_df['Start X'].apply(lambda x:np.round(x,4))
    full_df['Rounded Start Y'] = full_df['Start Y'].apply(lambda x:np.round(x,4))
    full_df['Orientation Error'] = full_df['Goal Orientation'] - full_df['End Orientation']
    full_df.to_pickle(folder +'combined_things.pkl')

    for file in episode_files:
        if 'combined' in file:
            print('we royally fucked up, everything sucks')
            assert 1==0
        os.remove(file)

if __name__ == '__main__':
    merge_from_folder('./data/Mothra_Full_Continue_New_weight/JA_S3/Test/')