import os
import subprocess
import cv2
import sys

def frame2vid(framedir, outdir, args):
    # Setup inpath / outpath
    print("-- Converting frames to mp4...")
    inpath = os.path.join(framedir, "frame")
    outpath = os.path.join(outdir, 'detection_movie.mp4')
    print(f"Reading from '{inpath}' \nWriting to '{outpath}' ")

    overwrite_str = '-y' if args.force else ''
    loglevel = 'verbose' if args.verbose else 'quiet'

    # Create command
    cmd = f"ffmpeg {overwrite_str} -loglevel {loglevel} "
    cmd += f'-framerate {args.FPS} -i {inpath}_%05d.png '
    cmd += f'-c:v libx264 -pix_fmt yuv420p {outpath}'
    cmd = [ele for ele in cmd.split(" ") if ele]

    # Run process
    process = subprocess.Popen(cmd)
    process.wait()
    print("Done.")

def vid2frames(src_video_path, raw_frames_dir, args):
    outpath = os.path.join(raw_frames_dir, 'frame')
    overwrite_str = '-y' if args.force else ''
    loglevel = 'verbose' if args.verbose else 'quiet'

    # Setup command
    cmd = f"ffmpeg {overwrite_str} -loglevel {loglevel} "

    cmd += f"-i {src_video_path} "

    if args.start_second is not None:
        cmd += f" -ss {args.start_second} "
    if args.end_second is not None:
        cmd += f"-to {args.end_second} "

    cmd += f"-vf fps={args.FPS} {outpath}_%05d.png"
    cmd = [ele for ele in cmd.split(" ") if ele]
    print("Running command: ", cmd)
    process = subprocess.Popen(cmd)
    process.wait()
    print("\nDone.")

# a simple version of vid2frames that uses cv2 instead of ffmpeg
def vid2frames_simple(src_video_path, raw_frames_dir, args):
    cap = cv2.VideoCapture(src_video_path)
    frame_count = 0

    # Read and save frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_path = os.path.join(raw_frames_dir, f'frame_{frame_count:05d}.png')
        if not cv2.imwrite(frame_path, frame):
            print("Could not write image to path", frame_path)

        frame_count += 1
        sys.stdout.write(f'\rProcessed frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()
