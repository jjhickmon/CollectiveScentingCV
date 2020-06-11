import os
import subprocess

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
    cmd += f'-c:v libx264 -s 1500x1500 -pix_fmt yuv420p {outpath}'
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
    process = subprocess.Popen(cmd)
    process.wait()
    print("\nDone.")
