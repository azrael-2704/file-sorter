import os
import time
import sys
import argparse

from file_utils import (
    display_directory_tree,
    collect_file_paths,
    separate_files_by_type,
    read_file_data
)

from data_processing import (
    compute_operations,
    execute_operations,
    process_files_by_date,
    process_files_by_type,
)

from text_processing import (
    process_text_files
)

from image_processing import (
    process_image_files
)

from output_filter import filter_specific_output  # Import the context manager
from local_ai import LocalVLMInference, LocalTextInference  # Import local AI implementations

def ensure_nltk_data():
    """Ensure that NLTK data is downloaded efficiently and quietly."""
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize models
image_inference = None
text_inference = None

def initialize_models():
    """Initialize the models if they haven't been initialized yet."""
    global image_inference, text_inference
    if image_inference is None or text_inference is None:
        # Initialize lightweight local inference models
        with filter_specific_output():
            # Initialize the image and text inference models
            image_inference = LocalVLMInference()
            text_inference = LocalTextInference()
            print("**----------------------------------------------**")
            print("**       Image inference model initialized      **")
            print("**       Text inference model initialized       **")
            print("**----------------------------------------------**")

def simulate_directory_tree(operations, base_path):
    """Simulate the directory tree based on the proposed operations."""
    tree = {}
    for op in operations:
        rel_path = os.path.relpath(op['destination'], base_path)
        parts = rel_path.split(os.sep)
        current_level = tree
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
    return tree

def print_simulated_tree(tree, prefix=''):
    """Print the simulated directory tree."""
    pointers = ['├── '] * (len(tree) - 1) + ['└── '] if tree else []
    for pointer, key in zip(pointers, tree):
        print(prefix + pointer + key)
        if tree[key]:  # If there are subdirectories or files
            extension = '│   ' if pointer == '├── ' else '    '
            print_simulated_tree(tree[key], prefix + extension)

def get_yes_no(prompt):
    """Prompt the user for a yes/no response."""
    while True:
        response = input(prompt).strip().lower()
        if response in ('yes', 'y'):
            return True
        elif response in ('no', 'n'):
            return False
        elif response == '/exit':
            print("Exiting program.")
            exit()
        else:
            print("Please enter 'yes' or 'no'. To exit, type '/exit'.")

def get_mode_selection():
    """Prompt the user to select a mode."""
    while True:
        print("Please choose the mode to organize your files:")
        print("1. By Content")
        print("2. By Date")
        print("3. By Type")
        response = input("Enter 1, 2, or 3 (or type '/exit' to exit): ").strip()
        if response == '/exit':
            print("Exiting program.")
            exit()
        elif response == '1':
            return 'content'
        elif response == '2':
            return 'date'
        elif response == '3':
            return 'type'
        else:
            print("Invalid selection. Please enter 1, 2, or 3. To exit, type '/exit'.")

def load_processed_files(log_path):
    """Load processed file paths from log file."""
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed.add(line.strip())
    return processed

def append_processed_files(log_path, files):
    """Append new processed file paths to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        for fp in files:
            f.write(fp + '\n')

def main():
    # If no command-line arguments provided, fall back to interactive prompts
    processed_log = 'processed_files.log'
    processed_files_set = load_processed_files(processed_log)

    if len(sys.argv) == 1:
        def get_mode_log_and_folder(mode):
            log_map = {
                'content': 'processed_content.log',
                'date': 'processed_date.log',
                'type': 'processed_type.log',
            }
            return log_map.get(mode, 'processed_content.log')

        def get_existing_folders_from_log(log_path):
            """Return set of folders used for processed files in log."""
            folders = set()
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        fp = line.strip()
                        if fp:
                            parent = os.path.basename(os.path.dirname(fp))
                            folders.add(parent)
            return folders

        ensure_nltk_data()

        print("-" * 50)
        print("**NOTE: Silent mode logs all outputs to a text file instead of displaying them in the terminal.")
        silent_mode = get_yes_no("Would you like to enable silent mode? (yes/no): ")
        log_file = 'operation_log.txt' if silent_mode else None

        while True:
            if not silent_mode:
                print("-" * 50)

            input_path = input("Enter the path of the directory you want to organize: ").strip()
            while not os.path.exists(input_path):
                message = f"Input path {input_path} does not exist. Please enter a valid path."
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(message + '\n')
                else:
                    print(message)
                input_path = input("Enter the path of the directory you want to organize: ").strip()

            message = f"Input path successfully uploaded: {input_path}"
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)

            output_path = input("Enter the path to store organized files and folders (press Enter to use 'organized_folder' in the input directory): ").strip()
            if not output_path:
                output_path = os.path.join(os.path.dirname(input_path), 'organized_folder')

            message = f"Output path successfully set to: {output_path}"
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)

            start_time = time.time()
            all_file_paths = collect_file_paths(input_path)
            # Filter out already processed files
            file_paths = [fp for fp in all_file_paths if fp not in processed_files_set]
            end_time = time.time()

            message = f"Time taken to load file paths: {end_time - start_time:.2f} seconds"
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)

            if not silent_mode:
                print("Directory tree before organizing:")
                display_directory_tree(input_path)

            if not file_paths:
                print("No new files to process. All files have already been processed.")
                continue

            # Mode selection loop
            while True:
                mode = get_mode_selection()

                if mode == 'content':
                    initialize_models()
                    # Ensure existing_folders is defined
                    try:
                        existing_folders
                    except NameError:
                        existing_folders = set()
                    image_files, text_files = separate_files_by_type(file_paths)
                    text_tuples = []
                    for fp in text_files:
                        text_content = read_file_data(fp)
                        if text_content is None:
                            msg = f"Unsupported or unreadable text file format: {fp}"
                            if log_file:
                                with open(log_file, 'a') as f:
                                    f.write(msg + '\n')
                            else:
                                print(msg)
                            continue
                        text_tuples.append((fp, text_content))
                    # Process text files first
                    data_texts = process_text_files(text_tuples, text_inference, silent=silent_mode, log_file=log_file, suggested_folders=existing_folders)
                    # Update folder suggestions with new folders from processed text files
                    text_folders = set(d['foldername'] for d in data_texts)
                    all_suggested_folders = existing_folders.union(text_folders)
                    # Now process images last, passing updated folder suggestions
                    data_images = process_image_files(image_files, image_inference, text_inference, silent=silent_mode, log_file=log_file, suggested_folders=all_suggested_folders)
                    all_data = data_texts + data_images
                    renamed_files = set()
                    processed_files = set()
                    operations = compute_operations(all_data, output_path, renamed_files, processed_files)

                elif mode == 'date':
                    operations = process_files_by_date(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
                elif mode == 'type':
                    operations = process_files_by_type(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
                else:
                    print("Invalid mode selected.")
                    continue

                # Show proposed structure
                if not silent_mode:
                    print("-" * 50)
                    print("Proposed directory structure:")
                    print(os.path.abspath(output_path))
                    simulated_tree = simulate_directory_tree(operations, output_path)
                    print_simulated_tree(simulated_tree)
                    print("-" * 50)
                else:
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write("Proposed directory structure logged.\n")

                proceed = get_yes_no("Would you like to proceed with these changes? (yes/no): ")
                if proceed:
                    os.makedirs(output_path, exist_ok=True)
                    execute_operations(operations, dry_run=False, silent=silent_mode, log_file=log_file)
                    if not silent_mode:
                        print("The files have been organized successfully.")
                    else:
                        if log_file:
                            with open(log_file, 'a') as f:
                                f.write("The files have been organized successfully.\n")
                    # After successful operation, append processed files
                    append_processed_files(processed_log, file_paths)
                    break
                else:
                    another_sort = get_yes_no("Would you like to choose another sorting method? (yes/no): ")
                    if another_sort:
                        continue
                    else:
                        print("Operation canceled by the user.")
                        break

            another_directory = get_yes_no("Would you like to organize another directory? (yes/no): ")
            if not another_directory:
                break
        return

    # CLI-driven mode (arguments provided)
    processed_log = 'processed_files.log'
    processed_files_set = load_processed_files(processed_log)
    parser = argparse.ArgumentParser(description="Organize files in a directory.")
    parser.add_argument("-i", "--input", help="Path of the directory to organize.")
    parser.add_argument("-o", "--output", help="Path to store organized files. Defaults to 'organized_folder' in the input directory's parent.")
    parser.add_argument("-m", "--mode", choices=['content', 'date', 'type'], help="The sorting mode to use.")
    parser.add_argument("-s", "--silent", action='store_true', help="Enable silent mode (logs to file).")
    parser.add_argument("-y", "--yes", action='store_true', help="Automatically answer 'yes' to all confirmation prompts.")

    args = parser.parse_args()

    # Prepare environment
    ensure_nltk_data()

    silent_mode = bool(args.silent)
    log_file = 'operation_log.txt' if silent_mode else None

    input_path = args.input.strip() if args.input else None
    if not input_path:
        print("Error: --input is required when using CLI mode. Run without args for interactive mode.")
        return

    output_path = args.output
    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path), 'organized_folder')

    # Start processing files
    start_time = time.time()
    all_file_paths = collect_file_paths(input_path)
    file_paths = [fp for fp in all_file_paths if fp not in processed_files_set]
    end_time = time.time()

    message = f"Time taken to load file paths: {end_time - start_time:.2f} seconds"
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    else:
        print(message)

    if not silent_mode:
        print("Directory tree before organizing:")
        display_directory_tree(input_path)

    if not file_paths:
        print("No new files to process. All files have already been processed.")
        return

    mode = args.mode
    operations = []

    if mode == 'content':
        # Initialize models once
        initialize_models()

        if not silent_mode:
            print("Processing files (content-based)... this may take a few minutes.")

        image_files, text_files = separate_files_by_type(file_paths)

        # Read text files
        text_tuples = []
        for fp in text_files:
            text_content = read_file_data(fp)
            if text_content is None:
                msg = f"Unsupported or unreadable text file format: {fp}"
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                else:
                    print(msg)
                continue
            text_tuples.append((fp, text_content))

        # Process text files first
        data_texts = process_text_files(text_tuples, text_inference, silent=silent_mode, log_file=log_file)
        # Then process images
        data_images = process_image_files(image_files, image_inference, text_inference, silent=silent_mode, log_file=log_file)

        all_data = data_texts + data_images
        renamed_files = set()
        processed_files = set()

        operations = compute_operations(all_data, output_path, renamed_files, processed_files)

    elif mode == 'date':
        operations = process_files_by_date(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
    elif mode == 'type':
        operations = process_files_by_type(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)

    # Show proposed structure
    if not silent_mode:
        print("-" * 50)
        print("Proposed directory structure:")
        print(os.path.abspath(output_path))
        simulated_tree = simulate_directory_tree(operations, output_path)
        print_simulated_tree(simulated_tree)
        print("-" * 50)
    else:
        if log_file:
            with open(log_file, 'a') as f:
                f.write("Proposed directory structure logged.\n")

    # Confirm and execute
    proceed = False
    if args.yes:
        proceed = True
    else:
        proceed = get_yes_no("Would you like to proceed with these changes? (yes/no): ")

    if proceed:
        os.makedirs(output_path, exist_ok=True)
        execute_operations(operations, dry_run=False, silent=silent_mode, log_file=log_file)
        # After successful operation, append processed files
        append_processed_files(processed_log, file_paths)
        if not silent_mode:
            print("The files have been organized successfully.")
        else:
            if log_file:
                with open(log_file, 'a') as f:
                    f.write("The files have been organized successfully.\n")
    else:
        if not silent_mode:
            print("Operation canceled.")
        else:
            if log_file:
                with open(log_file, 'a') as f:
                    f.write("Operation canceled by user.\n")


if __name__ == '__main__':
    main()