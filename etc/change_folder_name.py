import os

base_path = '/home/dabin/video2sound/CondFoleyGen/data/greatesthit/greatesthit-proccess-resized'

for folder_name in os.listdir(base_path):
    if '_denoised' in folder_name:
        new_folder_name = folder_name.replace('_denoised', '')
        original_folder_path = os.path.join(base_path, folder_name)
        new_folder_path = os.path.join(base_path, new_folder_name)

        os.rename(original_folder_path, new_folder_path)
        print(f'Renamed "{original_folder_path}" to "{new_folder_path}"')

print('Folder renaming process completed.')