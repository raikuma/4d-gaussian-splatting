import os
import sys

if __name__ == "__main__":
    model_path = sys.argv[1]
    iteration = sys.argv[2]

    file_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)
    file_size = os.path.getsize(file_path)
    with open(os.path.join(model_path, "storage.txt"), 'w') as f:
        f.write(f"File size: {file_size}\n")
        f.write(f"File size in MB: {file_size / (1024 * 1024):.2f}\n")
    print(f"File size: {file_size}")
    print(f"File size in MB: {file_size / (1024 * 1024):.2f}")