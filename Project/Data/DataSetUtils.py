import os
import shutil


def copy_all_files_flat(source_root, destination_folder):
    """
    Recursively copies all files from source_root (including subfolders)
    into destination_folder (flat structure).

    If duplicate filenames exist, a numeric suffix will be added.
    """

    # וידוא שתיקיית היעד קיימת
    os.makedirs(destination_folder, exist_ok=True)

    for root, dirs, files in os.walk(source_root):
        for filename in files:
            source_path = os.path.join(root, filename)
            destination_path = os.path.join(destination_folder, filename)

            # טיפול בהתנגשות שמות
            if os.path.exists(destination_path):
                name, ext = os.path.splitext(filename)
                counter = 1
                while True:
                    new_filename = f"{name}_{counter}{ext}"
                    new_destination = os.path.join(destination_folder, new_filename)
                    if not os.path.exists(new_destination):
                        destination_path = new_destination
                        break
                    counter += 1

            shutil.copy2(source_path, destination_path)

    print("Finished copying all files.")


if __name__ == "__main__":
    source_directory = r"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\dev-clean"
    destination_directory = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train"

    copy_all_files_flat(source_directory, destination_directory)
