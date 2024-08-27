import numpy as np


def main():
    gu = 10
    gv = 10

    eu = 10
    ev = 10

    gaze_vector = np.array([gu - eu, gv - ev])
    norm_gaze_vector = (
        1.0 if np.linalg.norm(gaze_vector) <= 0.0 else np.linalg.norm(gaze_vector)
    )
    print(norm_gaze_vector)


if __name__ == "__main__":
    main()
