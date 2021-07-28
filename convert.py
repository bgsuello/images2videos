import cv2
import numpy as np
import glob
import argparse
import re

from tqdm import tqdm
from subprocess import Popen, PIPE
from multiprocessing.pool import ThreadPool


numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def process_images(video_out, sequence, args):
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(
        *'DIVX'), args.fps, (args.width, args.height))
    for frame in tqdm(sequence, video_out):
        img = cv2.imread(frame)
        out.write(img)
    out.release()


def main():
    global main_progress

    parser = argparse.ArgumentParser(description='Convert images to videos')
    parser.add_argument(
        'images', type=str, help='images location glob pattern, ex: D:\\images\\*.jpg')
    parser.add_argument('--prefix', type=str,
                        help='sample out will produce out_<id>.avi')
    parser.add_argument('--width', default=800, type=int,
                        help='output video width')
    parser.add_argument('--height', default=600, type=int,
                        help='output video height')
    parser.add_argument('--fps', default=60, type=float,
                        help='output video fps')
    parser.add_argument('--ext', default='jpg', help='supported are jpg, png')
    parser.add_argument('--fpv', type=int, help='frames per video')
    parser.add_argument('--start', type=int, default=0,
                        help='starting count, default: 0')
    parser.add_argument('--end', type=int, default=0,
                        help='number of videos to output, default: 0(all)')
    parser.add_argument('-j', '--workers', dest='workers',
                        type=int, default=6, help='number of workers')
    parser.add_argument('-p', '--pad',
                        type=int, default=4, help='pad numbers with zeros unless specified, default to 04(four zeros)')

    args = parser.parse_args()

    images = sorted(glob.glob(args.images), key=numerical_sort)

    sequences = np.split(np.array(images), np.arange(
        args.fpv, len(images), args.fpv))

    workers = args.end if 0 < args.end < args.workers else args.workers

    print('Number of video sequence   :', len(sequences))
    print('Number of videos to output :', args.end if args.end > 0 else len(sequences))

    print(f'\n\nStarting request with {workers} workers...\n\n')

    pool = ThreadPool(workers)

    for index, sequence in enumerate(sequences):
        if args.end > 0 and index == args.end:
            break
        pool.apply_async(
            process_images, (f'{args.prefix}_{(index + args.start):0{args.pad}}.avi', sequence, args,))

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
