import pyrealsense2 as rs
import numpy as np
from mjpeg_streamer import MjpegServer, Stream
from dotenv import load_dotenv
import os

if __name__ == "__main__":
  load_dotenv()
  pipe = rs.pipeline()
  config = rs.config()
  
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  
  pipe.start(config)
  print("RealSense pipeline started")
  
  server = MjpegServer("0.0.0.0", 8080)
  stream = Stream("realsense", size=(640, 480), quality=85, fps=30)
  server.add_stream(stream)
  server.start()
  
  print("MJPEG server started on http://localhost:8080")
  print("Access stream at: http://localhost:8080")
  
  try:
    while True:
      # Wait for frames
      frames = pipe.wait_for_frames()
      color_frame = frames.get_color_frame()
      
      if not color_frame:
          continue
      
      # Convert to numpy array (already in BGR format)
      color_image = np.asanyarray(color_frame.get_data())
      
      # Push frame to stream
      stream.set_frame(color_image)
              
  except KeyboardInterrupt:
    print("\nStopping stream...")
  finally:
    server.stop()
    pipe.stop()
    print("Stopped")
