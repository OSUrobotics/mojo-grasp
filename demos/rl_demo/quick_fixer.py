import json
import os
import pathlib

root_name='data'
full_path = os.path.abspath('.')+'/data'
for folder in os.listdir(full_path):
    # print(folder)
    try:
        with open(full_path+'/'+folder+'/experiment_config.json', 'r') as file:
            config = json.load(file)
        save_path = full_path+'/'+folder+'/'
        if config['save_path'] == save_path:
            # print(save_path)
            pass
        else:
            print(config['save_path'], save_path)
            with open(full_path+'/'+folder+'/'+sub_folder+'/experiment_config.json','w') as file:
                json.dump(config,file)
    except FileNotFoundError:
        for sub_folder in os.listdir(full_path+'/'+folder):
            try:
                with open(full_path+'/'+folder+'/'+sub_folder+'/experiment_config.json','r') as file:
                    config = json.load(file)
                save_path = full_path+'/'+folder+'/'+sub_folder+'/'
                if config['save_path'] == save_path:
                    # print(save_path)
                    pass
                else:
                    print(config['save_path'], save_path)
                    config['save_path'] = save_path
                    with open(full_path+'/'+folder+'/'+sub_folder+'/experiment_config.json','w') as file:
                        json.dump(config,file)
            except NotADirectoryError:
                print('not running it, no experiment config', full_path+'/'+folder + '/' + sub_folder)
            except FileNotFoundError:
                print('not running it, no experiment config', full_path+'/'+folder + '/' + sub_folder)
# high_level_path =  pathlib.Path(confi'./'+root_name+'/'+folderg['save_path']).parent.resolve().parent.resolve().parent.resolve().parent.resolve().parent.resolve()
# high_level_path = str(high_level_path)