import sys
import numpy as np
import skvideo.io
import concurrent.futures
import time

def _detect_black_bars_from_video(frames, blackbar_threshold=16, max_perc_to_trim=.2):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars####################
    has_content = frames.max(axis=(0, -1)) >= blackbar_threshold
    h, w = has_content.shape

    y_frames = np.where(has_content.any(1))[0]
    if y_frames.size == 0:
        print("Oh no, there are no valid yframes")
        y_frames = [h // 2]

    y1 = min(y_frames[0], int(h * max_perc_to_trim))
    y2 = max(y_frames[-1] + 1, int(h * (1 - max_perc_to_trim)))

    x_frames = np.where(has_content.any(0))[0]
    if x_frames.size == 0:
        print("Oh no, there are no valid xframes")
        x_frames = [w // 2]
    x1 = min(x_frames[0], int(w * max_perc_to_trim))
    x2 = max(x_frames[-1] + 1, int(w * (1 - max_perc_to_trim)))
    return y1, y2, x1, x2


def extract_all_frames_from_video(video_file, blackbar_threshold=32, max_perc_to_trim=0.2,
                                  every_nth_frame=1, verbosity=0):
    """
    Same as exact_frames_from_video but no times meaning we grab every single frame
    :param video_file:
    :param r:
    :param blackbar_threshold:
    :param max_perc_to_trim:
    :return:
    """
    reader = skvideo.io.FFmpegReader(video_file, outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24'},
                                     verbosity=verbosity)

    # frames = [x for x in iter(reader.nextFrame())]
    frames = []
    for i, frame in enumerate(reader.nextFrame()):
        if (i % every_nth_frame) == 0:
            frames.append(frame)

    frames = np.stack(frames)
    y1, y2, x1, x2 = _detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
                                                   max_perc_to_trim=max_perc_to_trim)
    frames = frames[:, y1:y2, x1:x2]
    return frames


def extract_single_frame_from_video(video_file, t, verbosity=0):
    """
    Reads the video, seeks to the given second option
    :param video_file: input video file
    :param t: where 2 seek to
    :param use_rgb: True if use RGB, else BGR
    :return: the frame at that timestep.
    """
    timecode = '{:.3f}'.format(t)
    input_dict ={ '-ss': timecode, '-threads': '1',}
    reader = skvideo.io.FFmpegReader(video_file,
                                     inputdict=input_dict,
                                     outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                     verbosity=verbosity,
                                     )
    try:
        frame = next(iter(reader.nextFrame()))
    except StopIteration:
        frame = None
    return frame

def extract_frames_from_video(video_file, times, info, use_multithreading=False, use_rgb=True,
                              blackbar_threshold=32, max_perc_to_trim=.20, verbose=False):
    """
    Extracts multiple things from the video and even handles black bars

    :param video_file: what we are loading
    :param times: timestamps to use
    :param use_multithreading: Whether to use multithreading
    :param use_rgb whether to use RGB (default) or BGR
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """

    def _extract(i):
        return i, extract_single_frame_from_video(video_file, times[i], verbosity=10 if verbose else 0)

    time1 = time.time()

    if not use_multithreading:
        frames = [_extract(i)[1] for i in range(len(times))]
    else:
        frames = [None for t in times]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            submitted_threads = (executor.submit(_extract, i) for i in range(len(times)))
            for future in concurrent.futures.as_completed(submitted_threads):
                try:
                    i, img = future.result()
                    frames[i] = img
                except Exception as exc:
                    print("Oh no {}".format(str(exc)), flush=True)
    if verbose:
        print("Extracting frames from video, multithreading={} took {:.3f}".format(use_multithreading,
                                                                               time.time() - time1), flush=True)
    if any([x is None for x in frames]):
        print(f"Fail on {video_file}", flush=True)
        return None

    frames = np.stack(frames)
    y1, y2, x1, x2 = _detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
                                                   max_perc_to_trim=max_perc_to_trim)
    frames = frames[:, y1:y2, x1:x2]

    #############
    return frames