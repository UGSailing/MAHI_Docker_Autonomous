import time
import autonomous.main_sprint as sprint
import autonomous.main_figure8 as figure8

def run(mission_id: int) -> bool:
    if (mission_id == 0):
        figure8.run()
    if (mission_id == 1):
        # Slalom
        pass
    if (mission_id == 2):
        # Docking
        pass
    if (mission_id == 3):
        sprint.run()
        pass

    return True


if __name__ == "__main__":
    sprint.run()