///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Windows;
using System.Windows.Threading;
using System.Windows.Media.Imaging;
using System.IO;
using Microsoft.Win32;

// Internal libraries
using OpenCVWrappers;
using CppInterop;
using CppInterop.LandmarkDetector;
using CameraInterop;
using FaceAnalyser_Interop;
using System.Globalization;
using Microsoft.WindowsAPICodePack.Dialogs;

namespace OpenFaceOffline
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        // Timing for measuring FPS
        #region High-Resolution Timing
        static DateTime startTime;
        static Stopwatch sw = new Stopwatch();

        static MainWindow()
        {
            startTime = DateTime.Now;
            sw.Start();
        }

        public static DateTime CurrentTime
        {
            get { return startTime + sw.Elapsed; }
        }
        #endregion

        // -----------------------------------------------------------------
        // Members
        // -----------------------------------------------------------------

        Thread processing_thread;

        // Some members for displaying the results
        private Capture capture;
        private WriteableBitmap latest_img;
        private WriteableBitmap latest_aligned_face;
        private WriteableBitmap latest_HOG_descriptor;

        // Managing the running of the analysis system
        private volatile bool thread_running;
        private volatile bool thread_paused = false;
        // Allows for going forward in time step by step
        // Useful for visualising things
        private volatile int skip_frames = 0;

        FpsTracker processing_fps = new FpsTracker();

        volatile bool detectionSucceeding = false;

        volatile bool reset = false;

        // For tracking
        FaceModelParameters clnf_params;
        CLNF clnf_model;
        FaceAnalyserManaged face_analyser;

        // Recording parameters (default values)
        bool record_HOG = false; // HOG features extracted from face images
        bool record_aligned = false; // aligned face images
        bool record_tracked_vid = false;

        // Check wich things need to be recorded
        bool record_2D_landmarks = true;
        bool record_3D_landmarks = false;
        bool record_model_params = true;
        bool record_pose = true;
        bool record_AUs = true;
        bool record_gaze = true;

        // Visualisation options
        bool show_tracked_video = true;
        bool show_appearance = true;
        bool show_geometry = true;
        bool show_aus = true;

        int image_output_size = 112;

        // TODO classifiers converted to regressors

        // TODO indication that track is done        

        // The recording managers, TODO they should be all one
        StreamWriter output_features_file;

        // Where the recording is done (by default in a record directory, from where the application executed), TODO maybe the same folder as iput?
        String record_root = "./record";

        // For AU visualisation and output        
        List<String> au_class_names;
        List<String> au_reg_names;

        // For AU prediction
        bool dynamic_AU_shift = true;
        bool dynamic_AU_scale = false;
        bool use_dynamic_models = true;

        public MainWindow()
        {
            InitializeComponent();

            // Set the icon
            Uri iconUri = new Uri("logo1.ico", UriKind.RelativeOrAbsolute);
            this.Icon = BitmapFrame.Create(iconUri);

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 2000), (Action)(() =>
            {
                RecordAUCheckBox.IsChecked = record_AUs;
                RecordAlignedCheckBox.IsChecked = record_aligned;
                RecordTrackedVidCheckBox.IsChecked = record_tracked_vid;
                RecordHOGCheckBox.IsChecked = record_HOG;
                RecordGazeCheckBox.IsChecked = record_gaze;
                RecordLandmarks2DCheckBox.IsChecked = record_2D_landmarks;
                RecordLandmarks3DCheckBox.IsChecked = record_3D_landmarks;
                RecordParamsCheckBox.IsChecked = record_model_params;
                RecordPoseCheckBox.IsChecked = record_pose;

                UseDynamicModelsCheckBox.IsChecked = use_dynamic_models;
                UseDynamicScalingCheckBox.IsChecked = dynamic_AU_scale;
                UseDynamicShiftingCheckBox.IsChecked = dynamic_AU_shift;
            }));

            String root = AppDomain.CurrentDomain.BaseDirectory;

            clnf_params = new FaceModelParameters(root, false);
            clnf_model = new CLNF(clnf_params);
            face_analyser = new FaceAnalyserManaged(root, use_dynamic_models, image_output_size);

        }

        // ----------------------------------------------------------
        // Actual work gets done here

        // The main function call for processing images or video files
        private void ProcessingLoop(String[] filenames, int cam_id = -1, int width = -1, int height = -1, bool multi_face = false)
        {

            thread_running = true;

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
            {
                ResetButton.IsEnabled = true;
                PauseButton.IsEnabled = true;
                StopButton.IsEnabled = true;
            }));

            // Create the video capture and call the VideoLoop
            if (filenames != null)
            {
                clnf_params.optimiseForVideo();
                if (cam_id == -2)
                {
                    List<String> image_files_all = new List<string>();
                    foreach (string image_name in filenames)
                        image_files_all.Add(image_name);

                    // Loading an image sequence that represents a video                   
                    capture = new Capture(image_files_all);

                    if (capture.isOpened())
                    {
                        // Prepare recording if any based on the directory
                        String file_no_ext = System.IO.Path.GetDirectoryName(filenames[0]);
                        file_no_ext = System.IO.Path.GetFileName(file_no_ext);

                        SetupRecording(record_root, file_no_ext, capture.width, capture.height, record_2D_landmarks, record_2D_landmarks, record_model_params, record_pose, record_AUs, record_gaze);

                        // Start the actual processing                        
                        VideoLoop();

                        // Clear up the recording
                        StopRecording();

                    }
                    else
                    {
                        string messageBoxText = "Failed to open an image";
                        string caption = "Not valid file";
                        MessageBoxButton button = MessageBoxButton.OK;
                        MessageBoxImage icon = MessageBoxImage.Warning;

                        // Display message box
                        MessageBox.Show(messageBoxText, caption, button, icon);
                    }
                }
                else if (cam_id == -3)
                {
                    SetupImageMode();
                    clnf_params.optimiseForImages();
                    // Loading an image file (or a number of them)
                    foreach (string filename in filenames)
                    {
                        if (!thread_running)
                        {
                            continue;
                        }

                        capture = new Capture(filename);

                        if (capture.isOpened())
                        {
                            // Start the actual processing                        
                            ProcessImage();

                        }
                        else
                        {
                            string messageBoxText = "File is not an image or the decoder is not supported.";
                            string caption = "Not valid file";
                            MessageBoxButton button = MessageBoxButton.OK;
                            MessageBoxImage icon = MessageBoxImage.Warning;

                            // Display message box
                            MessageBox.Show(messageBoxText, caption, button, icon);
                        }
                    }
                }
                else
                {
                    clnf_params.optimiseForVideo();
                    // Loading a video file (or a number of them)
                    foreach (string filename in filenames)
                    {
                        if (!thread_running)
                        {
                            continue;
                        }

                        capture = new Capture(filename);

                        if (capture.isOpened())
                        {
                            // Prepare recording if any
                            String file_no_ext = System.IO.Path.GetFileNameWithoutExtension(filename);

                            SetupRecording(record_root, file_no_ext, capture.width, capture.height, record_2D_landmarks, record_3D_landmarks, record_model_params, record_pose, record_AUs, record_gaze);

                            // Start the actual processing                        
                            VideoLoop();

                            // Clear up the recording
                            StopRecording();
                        }
                        else
                        {
                            string messageBoxText = "File is not a video or the codec is not supported.";
                            string caption = "Not valid file";
                            MessageBoxButton button = MessageBoxButton.OK;
                            MessageBoxImage icon = MessageBoxImage.Warning;

                            // Display message box
                            MessageBox.Show(messageBoxText, caption, button, icon);
                        }
                    }
                }
            }

            // TODO this should be up a level
            // Some GUI clean up
            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
            {
                Console.WriteLine("Cleaning up after processing is done");
                PauseButton.IsEnabled = false;
                StopButton.IsEnabled = false;
                ResetButton.IsEnabled = false;
                NextFiveFramesButton.IsEnabled = false;
                NextFrameButton.IsEnabled = false;
            }));

        }

        // Capturing and processing the video frame by frame
        private void ProcessImage()
        {
            Thread.CurrentThread.IsBackground = true;

            clnf_model.Reset();
            face_analyser.Reset();


            //////////////////////////////////////////////
            // CAPTURE FRAME AND DETECT LANDMARKS FOLLOWED BY THE REQUIRED IMAGE PROCESSING
            //////////////////////////////////////////////
            RawImage frame = null;
            double progress = -1;

            frame = new RawImage(capture.GetNextFrame(false));
            progress = capture.GetProgress();

            if (frame.Width == 0)
            {
                // This indicates that we reached the end of the video file
                return;
            }

            var grayFrame = new RawImage(capture.GetCurrentFrameGray());

            if (grayFrame == null)
            {
                Console.WriteLine("Gray is empty");
                return;
            }

            List<List<Tuple<double, double>>> landmark_detections = ProcessImage(clnf_model, clnf_params, frame, grayFrame);

            List<Point> landmark_points = new List<Point>();

            for (int i = 0; i < landmark_detections.Count; ++i)
            {

                List<Tuple<double, double>> landmarks = landmark_detections[i];
                foreach (var p in landmarks)
                {
                    landmark_points.Add(new Point(p.Item1, p.Item2));
                }
            }

            // Visualisation
            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
            {
                if (show_tracked_video)
                {
                    if (latest_img == null)
                    {
                        latest_img = frame.CreateWriteableBitmap();
                    }

                    frame.UpdateWriteableBitmap(latest_img);

                    video.Source = latest_img;
                    video.Confidence = 1;
                    video.FPS = processing_fps.GetFPS();
                    video.Progress = progress;

                    video.OverlayLines = new List<Tuple<Point, Point>>();

                    video.OverlayPoints = landmark_points;
                }

            }));

            latest_img = null;
        }


        // Capturing and processing the video frame by frame
        private void VideoLoop()
        {
            Thread.CurrentThread.IsBackground = true;

            DateTime? startTime = CurrentTime;

            var lastFrameTime = CurrentTime;

            clnf_model.Reset();
            face_analyser.Reset();

            // TODO add an ability to change these through a calibration procedure or setting menu
            double fx, fy, cx, cy;
            fx = 500.0;
            fy = 500.0;
            cx = cy = -1;

            int frame_id = 0;

            double fps = capture.GetFPS();
            if (fps <= 0) fps = 30;

            while (thread_running)
            {
                //////////////////////////////////////////////
                // CAPTURE FRAME AND DETECT LANDMARKS FOLLOWED BY THE REQUIRED IMAGE PROCESSING
                //////////////////////////////////////////////
                RawImage frame = null;
                double progress = -1;

                frame = new RawImage(capture.GetNextFrame(false));
                progress = capture.GetProgress();

                if (frame.Width == 0)
                {
                    // This indicates that we reached the end of the video file
                    break;
                }

                // TODO stop button should actually clear the video
                lastFrameTime = CurrentTime;
                processing_fps.AddFrame();

                var grayFrame = new RawImage(capture.GetCurrentFrameGray());

                if (grayFrame == null)
                {
                    Console.WriteLine("Gray is empty");
                    continue;
                }

                // This is more ore less guess work, but seems to work well enough
                if (cx == -1)
                {
                    fx = fx * (grayFrame.Width / 640.0);
                    fy = fy * (grayFrame.Height / 480.0);

                    fx = (fx + fy) / 2.0;
                    fy = fx;

                    cx = grayFrame.Width / 2f;
                    cy = grayFrame.Height / 2f;
                }

                bool detectionSucceeding = ProcessFrame(clnf_model, clnf_params, frame, grayFrame, fx, fy, cx, cy);

                double scale = clnf_model.GetRigidParams()[0];

                double confidence = (-clnf_model.GetConfidence()) / 2.0 + 0.5;

                if (confidence < 0)
                    confidence = 0;
                else if (confidence > 1)
                    confidence = 1;

                List<double> pose = new List<double>();
                clnf_model.GetPose(pose, fx, fy, cx, cy);
                List<double> non_rigid_params = clnf_model.GetNonRigidParams();

                // The face analysis step (only done if recording AUs, HOGs or video)
                if (record_AUs || record_HOG || record_aligned || show_aus || show_appearance || record_tracked_vid || record_gaze)
                {
                    face_analyser.AddNextFrame(frame, clnf_model, fx, fy, cx, cy, false, show_appearance, record_tracked_vid);
                }

                List<Tuple<Point, Point>> lines = null;
                List<Tuple<double, double>> landmarks = null;
                List<Tuple<double, double>> eye_landmarks = null;
                List<Tuple<Point, Point>> gaze_lines = null;
                Tuple<double, double> gaze_angle = new Tuple<double, double>(0, 0);

                if (detectionSucceeding)
                {
                    landmarks = clnf_model.CalculateLandmarks();
                    eye_landmarks = clnf_model.CalculateEyeLandmarks();
                    lines = clnf_model.CalculateBox((float)fx, (float)fy, (float)cx, (float)cy);
                    gaze_lines = face_analyser.CalculateGazeLines(scale, (float)fx, (float)fy, (float)cx, (float)cy);
                    gaze_angle = face_analyser.GetGazeAngle();
                }

                // Visualisation
                Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
                {
                    if (show_aus)
                    {
                        var au_classes = face_analyser.GetCurrentAUsClass();
                        var au_regs = face_analyser.GetCurrentAUsReg();

                        auClassGraph.Update(au_classes);

                        var au_regs_scaled = new Dictionary<String, double>();
                        foreach (var au_reg in au_regs)
                        {
                            au_regs_scaled[au_reg.Key] = au_reg.Value / 5.0;
                            if (au_regs_scaled[au_reg.Key] < 0)
                                au_regs_scaled[au_reg.Key] = 0;

                            if (au_regs_scaled[au_reg.Key] > 1)
                                au_regs_scaled[au_reg.Key] = 1;
                        }
                        auRegGraph.Update(au_regs_scaled);
                    }

                    if (show_geometry)
                    {
                        int yaw = (int)(pose[4] * 180 / Math.PI + 0.5);
                        int roll = (int)(pose[5] * 180 / Math.PI + 0.5);
                        int pitch = (int)(pose[3] * 180 / Math.PI + 0.5);

                        YawLabel.Content = yaw + "°";
                        RollLabel.Content = roll + "°";
                        PitchLabel.Content = pitch + "°";

                        XPoseLabel.Content = (int)pose[0] + " mm";
                        YPoseLabel.Content = (int)pose[1] + " mm";
                        ZPoseLabel.Content = (int)pose[2] + " mm";

                        nonRigidGraph.Update(non_rigid_params);

                        // Update eye gaze
                        GazeXLabel.Content = gaze_angle.Item1 * (180.0 / Math.PI);
                        GazeYLabel.Content = gaze_angle.Item2 * (180.0 / Math.PI);

                    }

                    if (show_tracked_video)
                    {
                        if (latest_img == null)
                        {
                            latest_img = frame.CreateWriteableBitmap();
                        }

                        frame.UpdateWriteableBitmap(latest_img);

                        video.Source = latest_img;
                        video.Confidence = confidence;
                        video.FPS = processing_fps.GetFPS();
                        video.Progress = progress;

                        if (!detectionSucceeding)
                        {
                            video.OverlayLines.Clear();
                            video.OverlayPoints.Clear();
                            video.OverlayEyePoints.Clear();
                            video.GazeLines.Clear();
                        }
                        else
                        {
                            video.OverlayLines = lines;

                            List<Point> landmark_points = new List<Point>();
                            foreach (var p in landmarks)
                            {
                                landmark_points.Add(new Point(p.Item1, p.Item2));
                            }

                            List<Point> eye_landmark_points = new List<Point>();
                            foreach (var p in eye_landmarks)
                            {
                                eye_landmark_points.Add(new Point(p.Item1, p.Item2));
                            }


                            video.OverlayPoints = landmark_points;
                            video.OverlayEyePoints = eye_landmark_points;
                            video.GazeLines = gaze_lines;
                        }
                    }

                    if (show_appearance)
                    {
                        RawImage aligned_face = face_analyser.GetLatestAlignedFace();
                        RawImage hog_face = face_analyser.GetLatestHOGDescriptorVisualisation();

                        if (latest_aligned_face == null)
                        {
                            latest_aligned_face = aligned_face.CreateWriteableBitmap();
                            latest_HOG_descriptor = hog_face.CreateWriteableBitmap();
                        }

                        aligned_face.UpdateWriteableBitmap(latest_aligned_face);
                        hog_face.UpdateWriteableBitmap(latest_HOG_descriptor);

                        AlignedFace.Source = latest_aligned_face;
                        AlignedHOG.Source = latest_HOG_descriptor;
                    }
                }));

                // Recording the tracked model
                RecordFrame(clnf_model, detectionSucceeding, frame_id + 1, frame, grayFrame, ((double)frame_id) / fps,
                    record_2D_landmarks, record_2D_landmarks, record_model_params, record_pose, record_AUs, record_gaze, fx, fy, cx, cy);

                if (reset)
                {
                    clnf_model.Reset();
                    face_analyser.Reset();
                    reset = false;
                }

                while (thread_running & thread_paused && skip_frames == 0)
                {
                    Thread.Sleep(10);
                }

                frame_id++;

                if (skip_frames > 0)
                    skip_frames--;

            }

            latest_img = null;
            skip_frames = 0;

            // Unpause if it's paused
            if (thread_paused)
            {
                Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
                {
                    PauseButton_Click(null, null);
                }));
            }
        }

        private void StopTracking()
        {
            // First complete the running of the thread
            if (processing_thread != null)
            {
                // Tell the other thread to finish
                thread_running = false;
                processing_thread.Join();
            }
        }



        // ----------------------------------------------------------
        // Interacting with landmark detection and face analysis

        private bool ProcessFrame(CLNF clnf_model, FaceModelParameters clnf_params, RawImage frame, RawImage grayscale_frame, double fx, double fy, double cx, double cy)
        {
            detectionSucceeding = clnf_model.DetectLandmarksInVideo(grayscale_frame, clnf_params);
            return detectionSucceeding;

        }

        private List<List<Tuple<double, double>>> ProcessImage(CLNF clnf_model, FaceModelParameters clnf_params, RawImage frame, RawImage grayscale_frame)
        {
            List<List<Tuple<double, double>>> landmark_detections = clnf_model.DetectMultiFaceLandmarksInImage(grayscale_frame, clnf_params);
            return landmark_detections;

        }


        // ----------------------------------------------------------
        // Recording helpers (TODO simplify)

        private void SetupRecording(String root, String filename, int width, int height, bool output_2D_landmarks, bool output_3D_landmarks,
                                    bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze)
        {
            // Disallow changing recording settings when the recording starts, TODO move this up a bit
            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
            {
                RecordingMenu.IsEnabled = false;
                UseDynamicModelsCheckBox.IsEnabled = false;
            }));

            if (!System.IO.Directory.Exists(root))
            {
                System.IO.Directory.CreateDirectory(root);
            }

            output_features_file = new StreamWriter(root + "/" + filename + ".txt");
            output_features_file.Write("frame, timestamp, confidence, success");

            if (output_gaze)
                output_features_file.Write(", gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_2_z");

            if (output_pose)
                output_features_file.Write(", pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz");

            if (output_2D_landmarks)
            {
                for (int i = 0; i < clnf_model.GetNumPoints(); ++i)
                {
                    output_features_file.Write(", x_" + i);
                }
                for (int i = 0; i < clnf_model.GetNumPoints(); ++i)
                {
                    output_features_file.Write(", y_" + i);
                }
            }

            if (output_3D_landmarks)
            {

                for (int i = 0; i < clnf_model.GetNumPoints(); ++i)
                {
                    output_features_file.Write(", X_" + i);
                }
                for (int i = 0; i < clnf_model.GetNumPoints(); ++i)
                {
                    output_features_file.Write(", Y_" + i);
                }
                for (int i = 0; i < clnf_model.GetNumPoints(); ++i)
                {
                    output_features_file.Write(", Z_" + i);
                }
            }

            if (output_model_params)
            {
                output_features_file.Write(", p_scale, p_rx, p_ry, p_rz, p_tx, p_ty");
                for (int i = 0; i < clnf_model.GetNumModes(); ++i)
                {
                    output_features_file.Write(", p_" + i);
                }
            }

            if (output_AUs)
            {

                au_reg_names = face_analyser.GetRegActionUnitsNames();
                au_reg_names.Sort();
                foreach (var name in au_reg_names)
                {
                    output_features_file.Write(", " + name + "_r");
                }

                au_class_names = face_analyser.GetClassActionUnitsNames();
                au_class_names.Sort();
                foreach (var name in au_class_names)
                {
                    output_features_file.Write(", " + name + "_c");
                }

            }

            output_features_file.WriteLine();


            if (record_aligned)
            {
                String aligned_root = root + "/" + filename + "_aligned/";
                System.IO.Directory.CreateDirectory(aligned_root);
                face_analyser.SetupAlignedImageRecording(aligned_root);
            }

            if (record_tracked_vid)
            {
                String vid_loc = root + "/" + filename + ".avi";
                System.IO.Directory.CreateDirectory(root);
                face_analyser.SetupTrackingRecording(vid_loc, width, height, 30);
            }

            if (record_HOG)
            {
                String filename_HOG = root + "/" + filename + ".hog";
                face_analyser.SetupHOGRecording(filename_HOG);
            }

        }

        private void StopRecording()
        {
            if (output_features_file != null)
                output_features_file.Close();

            if (record_HOG)
                face_analyser.StopHOGRecording();

            if (record_tracked_vid)
                face_analyser.StopTrackingRecording();

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
            {
                RecordingMenu.IsEnabled = true;
                UseDynamicModelsCheckBox.IsEnabled = true;

            }));

        }

        // Recording the relevant objects
        private void RecordFrame(CLNF clnf_model, bool success, int frame_ind, RawImage frame, RawImage grayscale_frame, double time_stamp, bool output_2D_landmarks, bool output_3D_landmarks,
                                    bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze, double fx, double fy, double cx, double cy)
        {
            // Making sure that full stop is used instead of a comma for data recording
            System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";

            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;

            double confidence = (-clnf_model.GetConfidence()) / 2.0 + 0.5;

            List<double> pose = new List<double>();
            clnf_model.GetPose(pose, fx, fy, cx, cy);

            output_features_file.Write(String.Format("{0}, {1}, {2:F3}, {3}", frame_ind, time_stamp, confidence, success ? 1 : 0));

            if (output_gaze)
            {
                var gaze = face_analyser.GetGazeCamera();
                var gaze_angle = face_analyser.GetGazeAngle();

                output_features_file.Write(String.Format(", {0:F5}, {1:F5}, {2:F5}, {3:F5}, {4:F5}, {5:F5}, {6:F5}, {7:F5}", gaze.Item1.Item1, gaze.Item1.Item2, gaze.Item1.Item3,
                    gaze.Item2.Item1, gaze.Item2.Item2, gaze.Item2.Item3, gaze_angle.Item1, gaze_angle.Item2));
            }

            if (output_pose)
                output_features_file.Write(String.Format(", {0:F3}, {1:F3}, {2:F3}, {3:F3}, {4:F3}, {5:F3}", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]));

            if (output_2D_landmarks)
            {
                List<Tuple<double, double>> landmarks_2d = clnf_model.CalculateLandmarks();

                for (int i = 0; i < landmarks_2d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_2d[i].Item1);

                for (int i = 0; i < landmarks_2d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_2d[i].Item2);
            }

            if (output_3D_landmarks)
            {
                List<System.Windows.Media.Media3D.Point3D> landmarks_3d = clnf_model.Calculate3DLandmarks(fx, fy, cx, cy);

                for (int i = 0; i < landmarks_3d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_3d[i].X);

                for (int i = 0; i < landmarks_3d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_3d[i].Y);

                for (int i = 0; i < landmarks_3d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_3d[i].Z);
            }

            if (output_model_params)
            {
                List<double> all_params = clnf_model.GetParams();

                for (int i = 0; i < all_params.Count; ++i)
                    output_features_file.Write(String.Format(", {0,0:F5}", all_params[i]));
            }

            if (output_AUs)
            {
                var au_regs = face_analyser.GetCurrentAUsReg();

                foreach (var name_reg in au_reg_names)
                    output_features_file.Write(", {0:F2}", au_regs[name_reg]);

                var au_classes = face_analyser.GetCurrentAUsClass();

                foreach (var name_class in au_class_names)
                    output_features_file.Write(", {0:F0}", au_classes[name_class]);

            }

            output_features_file.WriteLine();

            if (record_aligned)
            {
                face_analyser.RecordAlignedFrame(frame_ind);
            }

            if (record_HOG)
            {
                face_analyser.RecordHOGFrame();
            }

            if (record_tracked_vid)
            {
                face_analyser.RecordTrackedFace();
            }
        }


        // ----------------------------------------------------------
        // Mode handling (image, video)
        // ----------------------------------------------------------
        private void SetupImageMode()
        {
            // Turn off recording
            record_aligned = false;
            record_HOG = false;
            record_tracked_vid = false;

            // Turn off unneeded visualisations
            show_tracked_video = true;
            show_appearance = false;
            show_geometry = false;
            show_aus = false;

            // Actually update the GUI accordingly
            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 2000), (Action)(() =>
            {
                RecordAUCheckBox.IsChecked = record_AUs;
                RecordAlignedCheckBox.IsChecked = record_aligned;
                RecordTrackedVidCheckBox.IsChecked = record_tracked_vid;
                RecordHOGCheckBox.IsChecked = record_HOG;
                RecordGazeCheckBox.IsChecked = record_gaze;
                RecordLandmarks2DCheckBox.IsChecked = record_2D_landmarks;
                RecordLandmarks3DCheckBox.IsChecked = record_3D_landmarks;
                RecordParamsCheckBox.IsChecked = record_model_params;
                RecordPoseCheckBox.IsChecked = record_pose;

                ShowVideoCheckBox.IsChecked = true;
                ShowAppearanceFeaturesCheckBox.IsChecked = false;
                ShowGeometryFeaturesCheckBox.IsChecked = false;
                ShowAUsCheckBox.IsChecked = false;

                VisualisationCheckBox_Click(null, null);
            }));

            // TODO change what next and back buttons do?
        }


        // ----------------------------------------------------------
        // Opening Videos/Images
        // ----------------------------------------------------------

        private void videoFileOpenClick(object sender, RoutedEventArgs e)
        {
            new Thread(() => openVideoFile()).Start();
        }

        private void openVideoFile()
        {
            StopTracking();

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 2, 0), (Action)(() =>
            {
                var d = new OpenFileDialog();
                d.Multiselect = true;
                d.Filter = "Video files|*.avi;*.wmv;*.mov;*.mpg;*.mpeg;*.mp4";

                if (d.ShowDialog(this) == true)
                {

                    string[] video_files = d.FileNames;

                    processing_thread = new Thread(() => ProcessingLoop(video_files));
                    processing_thread.Start();

                }
            }));
        }


        private void imageFileOpenClick(object sender, RoutedEventArgs e)
        {
            new Thread(() => imageOpen()).Start();
        }

        private void imageOpen()
        {
            StopTracking();

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 2, 0), (Action)(() =>
            {
                var d = new OpenFileDialog();
                d.Multiselect = true;
                d.Filter = "Image files|*.jpg;*.jpeg;*.bmp;*.png;*.gif";

                if (d.ShowDialog(this) == true)
                {

                    string[] image_files = d.FileNames;

                    processing_thread = new Thread(() => ProcessingLoop(image_files, -3));
                    processing_thread.Start();

                }
            }));
        }

        private void imageSequenceFileOpenClick(object sender, RoutedEventArgs e)
        {
            new Thread(() => imageSequenceOpen()).Start();
        }

        private void imageSequenceOpen()
        {
            StopTracking();

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 2, 0), (Action)(() =>
            {
                var d = new OpenFileDialog();
                d.Multiselect = true;
                d.Filter = "Image files|*.jpg;*.jpeg;*.bmp;*.png;*.gif";

                if (d.ShowDialog(this) == true)
                {

                    string[] image_files = d.FileNames;

                    processing_thread = new Thread(() => ProcessingLoop(image_files, -2));
                    processing_thread.Start();

                }
            }));
        }

        // --------------------------------------------------------
        // Button handling
        // --------------------------------------------------------

        // Cleanup stuff when closing the window
        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (processing_thread != null)
            {
                // Stop capture and tracking
                thread_running = false;
                processing_thread.Join();

                capture.Dispose();
            }
            face_analyser.Dispose();
        }

        // Stopping the tracking
        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            if (processing_thread != null)
            {
                // Stop capture and tracking
                thread_paused = false;
                thread_running = false;
                processing_thread.Join();

                PauseButton.IsEnabled = false;
                NextFrameButton.IsEnabled = false;
                NextFiveFramesButton.IsEnabled = false;
                StopButton.IsEnabled = false;
                ResetButton.IsEnabled = false;
                RecordingMenu.IsEnabled = true;

                UseDynamicModelsCheckBox.IsEnabled = true;
            }
        }

        // Resetting the tracker
        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            if (processing_thread != null)
            {
                // Stop capture and tracking
                reset = true;
            }
        }

        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            if (processing_thread != null)
            {
                // Stop capture and tracking                
                thread_paused = !thread_paused;

                ResetButton.IsEnabled = !thread_paused;

                NextFrameButton.IsEnabled = thread_paused;
                NextFiveFramesButton.IsEnabled = thread_paused;

                if (thread_paused)
                {
                    PauseButton.Content = "Resume";
                }
                else
                {
                    PauseButton.Content = "Pause";
                }
            }
        }

        private void SkipButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender.Equals(NextFrameButton))
            {
                skip_frames += 1;
            }
            else if (sender.Equals(NextFiveFramesButton))
            {
                skip_frames += 5;
            }
        }


        private void VisualisationCheckBox_Click(object sender, RoutedEventArgs e)
        {
            show_tracked_video = ShowVideoCheckBox.IsChecked;
            show_appearance = ShowAppearanceFeaturesCheckBox.IsChecked;
            show_geometry = ShowGeometryFeaturesCheckBox.IsChecked;
            show_aus = ShowAUsCheckBox.IsChecked;

            // Collapsing or restoring the windows here
            if (!show_tracked_video)
            {
                VideoBorder.Visibility = System.Windows.Visibility.Collapsed;
                MainGrid.ColumnDefinitions[0].Width = new GridLength(0, GridUnitType.Star);
            }
            else
            {
                VideoBorder.Visibility = System.Windows.Visibility.Visible;
                MainGrid.ColumnDefinitions[0].Width = new GridLength(2.1, GridUnitType.Star);
            }

            if (!show_appearance)
            {
                AppearanceBorder.Visibility = System.Windows.Visibility.Collapsed;
                MainGrid.ColumnDefinitions[1].Width = new GridLength(0, GridUnitType.Star);
            }
            else
            {
                AppearanceBorder.Visibility = System.Windows.Visibility.Visible;
                MainGrid.ColumnDefinitions[1].Width = new GridLength(0.8, GridUnitType.Star);
            }

            // Collapsing or restoring the windows here
            if (!show_geometry)
            {
                GeometryBorder.Visibility = System.Windows.Visibility.Collapsed;
                MainGrid.ColumnDefinitions[2].Width = new GridLength(0, GridUnitType.Star);
            }
            else
            {
                GeometryBorder.Visibility = System.Windows.Visibility.Visible;
                MainGrid.ColumnDefinitions[2].Width = new GridLength(1.0, GridUnitType.Star);
            }

            // Collapsing or restoring the windows here
            if (!show_aus)
            {
                ActionUnitBorder.Visibility = System.Windows.Visibility.Collapsed;
                MainGrid.ColumnDefinitions[3].Width = new GridLength(0, GridUnitType.Star);
            }
            else
            {
                ActionUnitBorder.Visibility = System.Windows.Visibility.Visible;
                MainGrid.ColumnDefinitions[3].Width = new GridLength(1.6, GridUnitType.Star);
            }

        }


        private void recordCheckBox_click(object sender, RoutedEventArgs e)
        {
            record_AUs = RecordAUCheckBox.IsChecked;
            record_aligned = RecordAlignedCheckBox.IsChecked;
            record_HOG = RecordHOGCheckBox.IsChecked;
            record_gaze = RecordGazeCheckBox.IsChecked;
            record_tracked_vid = RecordTrackedVidCheckBox.IsChecked;
            record_2D_landmarks = RecordLandmarks2DCheckBox.IsChecked;
            record_3D_landmarks = RecordLandmarks3DCheckBox.IsChecked;
            record_model_params = RecordParamsCheckBox.IsChecked;
            record_pose = RecordPoseCheckBox.IsChecked;
        }

        private void UseDynamicModelsCheckBox_Click(object sender, RoutedEventArgs e)
        {
            dynamic_AU_shift = UseDynamicShiftingCheckBox.IsChecked;
            dynamic_AU_scale = UseDynamicScalingCheckBox.IsChecked;

            if (use_dynamic_models != UseDynamicModelsCheckBox.IsChecked)
            {
                // Change the face analyser, this should be safe as the model is only allowed to change when not running
                String root = AppDomain.CurrentDomain.BaseDirectory;
                face_analyser = new FaceAnalyserManaged(root, UseDynamicModelsCheckBox.IsChecked, image_output_size);
            }
            use_dynamic_models = UseDynamicModelsCheckBox.IsChecked;
        }

        private void setOutputImageSize_Click(object sender, RoutedEventArgs e)
        {

            NumberEntryWindow number_entry_window = new NumberEntryWindow();
            number_entry_window.Icon = this.Icon;

            number_entry_window.WindowStartupLocation = WindowStartupLocation.CenterScreen;

            if (number_entry_window.ShowDialog() == true)
            {
                image_output_size = number_entry_window.OutputInt;
                String root = AppDomain.CurrentDomain.BaseDirectory;
                face_analyser = new FaceAnalyserManaged(root, use_dynamic_models, image_output_size);

            }
        }

        private void OutputLocationItem_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new CommonOpenFileDialog();
            dlg.Title = "Select output directory";
            dlg.IsFolderPicker = true;
            dlg.AllowNonFileSystemItems = false;
            dlg.EnsureFileExists = true;
            dlg.EnsurePathExists = true;
            dlg.EnsureReadOnly = false;
            dlg.EnsureValidNames = true;
            dlg.Multiselect = false;
            dlg.ShowPlacesList = true;

            if (dlg.ShowDialog() == CommonFileDialogResult.Ok)
            {
                var folder = dlg.FileName;
                record_root = folder;
            }
        }
    }
}
