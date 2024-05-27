import concurrent.futures
import argparse

from preprocess_futures import process_file

from typing import Tuple


def process_args() -> Tuple[str, int]:
    parser = argparse.ArgumentParser(description='manage multicore threading for preprocess_futures',
                                     prog='futureRunner.py')
    parser.add_argument('file', help='file containing the names of the dataset files')
    parser.add_argument('workers', help='set number of cores, (set to 24 on lxblade)')

    args = parser.parse_args()
    print(args)
    file_: str = args.file
    workers_: int = args.workers
    return file_, workers_

def read_filenames(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def main(filenames_path, workers=4):
    filenames = read_filenames(filenames_path)#[:8]
    total_files = len(filenames)
    completed = 0  # Initialize a counter for completed tasks

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, filename): filename for filename in filenames}
        
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            completed += 1  # Increment the counter when a task is completed
            try:
                future.result()
                print(f'INFO: {filename} processed successfully. [{completed}/{total_files} completed]')
            except Exception as exc:
                print(f'ERROR: {filename} generated an exception: {exc}. [{completed}/{total_files} completed]')


if __name__ == '__main__':
    filenames_path, workers = process_args()
    main(filenames_path, workers=workers)
