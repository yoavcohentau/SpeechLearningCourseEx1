import os
import json
import argparse


def build_json_from_folder(folder_path):
    """
    Returns sorted list of wav filenames without extension
    """
    files = []

    for f in os.listdir(folder_path):
        if f.endswith(".wav"):
            name = os.path.splitext(f)[0]
            files.append(name)

    files.sort()
    return files


def save_json(list_data, output_path):
    with open(output_path, "w") as f:
        json.dump(list_data, f, indent=2)
    print(f"Saved: {output_path} ({len(list_data)} entries)")


def main(dataset_root):

    mix_dir = os.path.join(dataset_root, "mixture")
    bf_dir = os.path.join(dataset_root, "mvdr_out")
    target_dir = os.path.join(dataset_root, "clean")

    assert os.path.exists(mix_dir), f"{mix_dir} not found"
    assert os.path.exists(bf_dir), f"{bf_dir} not found"
    assert os.path.exists(target_dir), f"{target_dir} not found"

    mix_list = build_json_from_folder(mix_dir)
    bf_list = build_json_from_folder(bf_dir)
    target_list = build_json_from_folder(target_dir)

    # sanity check lengths
    print("mix:", len(mix_list))
    print("bf:", len(bf_list))
    print("target:", len(target_list))

    # save
    save_json(mix_list, os.path.join(dataset_root, "mix.json"))
    save_json(bf_list, os.path.join(dataset_root, "bf.json"))
    save_json(target_list, os.path.join(dataset_root, "target.json"))

    print("\nDone.")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     required=True,
    #     help="Path to dataset root containing mix/, bf/, target/"
    # )
    #
    # args = parser.parse_args()

    # main(args.dataset_root)

    main(r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_valid")
