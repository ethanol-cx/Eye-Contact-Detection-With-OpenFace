# Ready to run eye detection software on Mac
In this folder you'll find a binary file named `DetectEyeContact` –– it is a modified version of `FaceLandmarkVid`, recompiled on a Mac to extend the functionality of detecting eye contact in babies. You'll need to download the [OpenFace model folder located here](https://www.dropbox.com/sh/f84ylup6npucn5t/AABzpvGSvE5E2GksPd7Yx5z_a?dl=0) before running `DetectEyeContact`. The `model` folder contains all of the mac-compiled models that OpenFace uses to extract facial features – download & place it in this directory prior to running.


## Instructions
Before running the binary, you'll need to start the python server that loads the prediction model that will be used by `DetectEyeContact`.
```
python3 predictorServer.py
```

Next, open up a second terminal window and navigate to this same directory.

While the `predictorServer.py` is running in another terminal window, run eye contact detection on a pre-recorded video by running
```
./DetectEyeContact -f video.mp4
```

To run eye contact detection on real-time video input (such as webcam), run
```
./DetectEyeContact -device 0
```
## Other notes
You can get some degree of customization without needing to recompile the C++.

For example, if you'd like to check against annotations while running a pre-recorded video, open the `predictorClient.py` file in a text-editor of your choice, and un-comment the corresponding blob.

Similarly, to write predictions to a file in real-time, open the `predictorClient.py` file and un-comment the corresponding blob.
