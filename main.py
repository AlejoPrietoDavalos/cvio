from pathlib import Path

from cvio.webcam import WebcamYOLO

def main():
    path_data = Path("data")
    path_data.mkdir(exist_ok=True)

    path_captures = path_data / "captures"
    path_captures.mkdir(exist_ok=True)

    vu = WebcamYOLO(path_captures)
    vu.run_webcam()

if __name__ == "__main__":
    main()
