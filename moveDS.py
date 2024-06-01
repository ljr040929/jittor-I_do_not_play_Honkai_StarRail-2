import os

def delete_ds_store_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f'Removed: {file_path}')
                except Exception as e:
                    print(f'Error removing {file_path}: {e}')


directory_to_clean = '/root/jittor-I_do_not_play_Honkai_StarRail-2/A'

delete_ds_store_files(directory_to_clean)
