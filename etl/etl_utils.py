
import os


def get_or_make_source_dirs(source_type: str, show_key: str = None) -> tuple[str, str]:
    dir_root = 'source'
    if not os.path.isdir(dir_root):
        os.mkdir(dir_root)
    dir_path = f'{dir_root}/{source_type}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if show_key:
        dir_path = f'{dir_path}/{show_key}'
    backup_dir_path = f'{dir_path}/backup'
    if not os.path.isdir(backup_dir_path):
        os.mkdir(backup_dir_path)

    return dir_path, backup_dir_path