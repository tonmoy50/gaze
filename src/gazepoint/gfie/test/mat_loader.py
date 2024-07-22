from scipy.io import loadmat
import os

CUR_DIR = os.path.join(os.path.dirname(__file__))
print(CUR_DIR)


def main():
    mat = loadmat(os.path.join(CUR_DIR, "metadata.mat"))
    print(mat.keys())
    # print(mat["gaze_dir"])

    


if __name__ == "__main__":
    main()
