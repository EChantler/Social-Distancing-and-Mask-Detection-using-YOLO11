# export the first 10 seconds of a video
from moviepy import VideoFileClip

# Load the video file
video_path = "C:\\Users\\ercha\\Downloads\\archive\\town.avi"
video = VideoFileClip(video_path)

# Extract the first 10 seconds
video_10s = video.subclipped(0, 10)

# Export the result to a new file
output_path = 'sample_video.mp4'
video_10s.write_videofile(output_path, codec='libx264')
