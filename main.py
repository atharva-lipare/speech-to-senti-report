# USAGE: python main.py <\path\to\input\video\folder> <\path\to\output\report\folder>
# example: python main.py C:\Users\Atharva\PycharmProjects\speech-to-senti-report\testDir C:\Users\Atharva\PycharmProjects\speech-to-senti-report\report_dir
import sys
import time
import glob
import os
from helper_functions import start_analysis, get_report
currentFilesCount = 0


def main():
    if len(sys.argv) != 3:
        print("Incorrect arguments. Please try again")
        sys.exit(0)
    media_dir = sys.argv[1]
    report_dir = sys.argv[2]
    print('MEDIA_DIR: ' + media_dir)
    print('REPORT_DIR: ' + report_dir)

    global currentFilesCount
    currentFilesCount = len(glob.glob(os.path.join(media_dir, '*')))
    print(currentFilesCount)
    while True:
        list_of_files = glob.glob(os.path.join(media_dir, '*'))
        num_files = len(list_of_files)
        if currentFilesCount != num_files:
            currentFilesCount = num_files
            if num_files == 0:
                continue
            latest_file = max(list_of_files, key=os.path.getctime)
            print(latest_file)

            # Beneath loop checks whether file is done writing/downloading which was written for WINDOWS platform
            while True:
                try:
                    with open(latest_file, 'rb') as _:
                        break
                except IOError:
                    time.sleep(3)

            print('Analysis started')
            start_analysis(latest_file, report_dir=report_dir)
        time.sleep(1)


def main1():
    get_report()


if __name__ == '__main__':
    main()
    #main1()
