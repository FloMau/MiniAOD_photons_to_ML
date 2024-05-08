import concurrent.futures
from preprocess_futures import process_file

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
    filenames_path = '/home/home1/institut_3a/kappe/CMSSW_12_6_0/src/MiniAOD_photons_to_ML/jobs/datafiles_low_pt.txt'
    # filenames_path = '/home/home1/institut_3a/kappe/CMSSW_12_6_0/src/MiniAOD_photons_to_ML/testfile.txt'
    main(filenames_path, workers=24)
