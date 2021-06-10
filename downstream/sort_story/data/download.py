import argparse
import os
import subprocess

from google_drive_downloader import GoogleDriveDownloader as gdd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--working_dir',
        default='data')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)

    if not os.path.exists('{}/SIS-with-labels.tar.gz'.format(args.working_dir)):
        subprocess.call('wget https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz -O {}/SIS-with-labels.tar.gz'.format(args.working_dir), shell=True)
    if not os.path.exists('{}/sis/test.story-in-sequence.json'.format(args.working_dir)):
        subprocess.call('tar -xvf {}/SIS-with-labels.tar.gz -C {}/'.format(args.working_dir, args.working_dir), shell=True)


    gdd.download_file_from_google_drive(file_id='0ByQS_kT8kViSWmtRa1lMcG1EaHc',
                                        dest_path='./data/val_images.tar.gz',
                                        unzip=False,
                                        showsize=True)

    if not os.path.exists('{}/images/val/2021867.jpg'.format(args.working_dir)):
        subprocess.call('tar -xvf {}/val_images.tar.gz -C {}/'.format(args.working_dir, args.working_dir), shell=True)
    
    gdd.download_file_from_google_drive(file_id='0ByQS_kT8kViSTHJ0cGxSVW1SRFk',
                                        dest_path='./data/test_images.tar.gz',
                                        showsize=True,
                                        unzip=False)

    if not os.path.exists('{}/test/9954863.jpg'.format(args.working_dir)):
        subprocess.call('tar -xvf {}/test_images.tar.gz -C {}/'.format(args.working_dir, args.working_dir), shell=True)


    
if __name__ == '__main__':
    main()
