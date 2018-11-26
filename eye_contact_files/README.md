# Eye Detection Source Code
This directory contains all of the resources you need to recompile the source code from scratch (in the [bin_replacement](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/tree/master/eye_contact_files/bin_replacement) folder) as well as easily run the software without needing to compile (in the [ready2run_mac_binary](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/tree/master/eye_contact_files/ready2run_mac_binary) folder).
## Easy run on Mac
Navigate to the [ready2run_mac_binary](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/tree/master/eye_contact_files/ready2run_mac_binary) folder to run the executable code on Mac. This won't work on linux.
## Source Code Installation and Recompilation
To have full control over the source code (and be able to recompile it) you'll need to run all of the [OpenFace installation instructions](https://github.com/TadasBaltrusaitis/OpenFace/wiki) in the root directory, and then replace the `FaceLandmarkVid.cpp` file for the one our team wrote, located in the [bin_replacement](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/tree/master/eye_contact_files/bin_replacement) folder. Once you've successfully replaced the OpenFace `FaceLandmarkVid.cpp` for ours, navigate back to the root directory and recompile using
```cmake```
and
```make```
Once everything compiles, you should have a bin folder. Then, go back to [eye_contact_files](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/tree/master/eye_contact_files) and copy the  bin_replacement/[predictorClient.py](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/blob/master/eye_contact_files/bin_replacement/predictorClient.py),  bin_replacement/[predictorServer.py](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/blob/master/eye_contact_files/bin_replacement/predictorServer.py), and bin_replacement/[dt.joblib](https://github.com/pashpashpash/Eye-Detection-With-OpenFace/blob/master/eye_contact_files/bin_replacement/dt.joblib) files into your newly made /bin/ folder inside of the root directory. Now you should have the compiled EyeDetection executable in `/bin/FaceLandmarkVid`, the two python files it calls as well as the prediction model `dt.joblib` file all inside of /bin/.

## Running Instructions
Once in the `/bin/` folder,  you'll need to start the python server that loads the prediction model that will be used by `FaceLandmarkVid` before running the binary,.
```
python3 predictorServer.py
```

If you get any errors, run a pip install on the missing dependencies. Next, open up a second terminal window and navigate to this same directory.

While the `predictorServer.py` is running in another terminal window, run eye contact detection on a pre-recorded video by running
```
./FaceLandmarkVid -f video.mp4
```

To run eye contact detection on real-time video input (such as webcam), run
```
./FaceLandmarkVid -device 0
```
## Other notes
You can get some degree of customization without needing to recompile the C++.

For example, if you'd like to check against annotations while running a pre-recorded video, open the `predictorClient.py` file in a text-editor of your choice, and un-comment the corresponding blob.

Similarly, to write predictions to a file in real-time, open the `predictorClient.py` file and un-comment the corresponding blob.
