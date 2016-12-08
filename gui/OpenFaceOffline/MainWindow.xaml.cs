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
        Recorder recorder;

        public bool RecordAligned { get; set; } // Aligned face images
        public bool RecordHOG { get; set; } // HOG features extracted from face images
        public bool Record2DLandmarks { get; set; } // 2D locations of facial landmarks (in pixels)
        public bool Record3DLandmarks { get; set; } // 3D locations of facial landmarks (in pixels)
        public bool RecordModelParameters { get; set; } // Facial shape parameters (rigid and non-rigid geometry)
        public bool RecordPose { get; set; } // Head pose (position and orientation)
        public bool RecordAUs { get; set; } // Facial action units
        public bool RecordGaze { get; set; } // Eye gaze

        // Visualisation options
        public bool ShowTrackedVideo { get; set; } // Eye gaze
        public bool ShowAppearance { get; set; } // Eye gaze
        public bool ShowGeometry { get; set; } // Eye gaze
        public bool ShowAUs { get; set; } // Eye gaze

        int image_output_size = 112;

        // TODO classifiers converted to regressors

        // TODO indication that track is done        
        
        // Where the recording is done (by default in a record directory, from where the application executed), TODO maybe the same folder as iput?
        String record_root = "./record";
        
        // For AU prediction
        public bool DynamicAUModels { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            this.DataContext = this; // For WPF data binding

            // Set the icon
            Uri iconUri = new Uri("logo1.ico", UriKind.RelativeOrAbsolute);
            this.Icon = BitmapFrame.Create(iconUri);

            // Setup the default features that will be recorded
            Record2DLandmarks = true; Record3DLandmarks = true; RecordModelParameters = true; RecordModelParameters = true; 
            RecordGaze = true; RecordAUs = true; RecordPose = true;
            RecordAligned = false; RecordHOG = false;

            ShowTrackedVideo = true;
            ShowAppearance = true;
            ShowGeometry = true;
            ShowAUs = true;

            DynamicAUModels = true;

            String root = AppDomain.CurrentDomain.BaseDirectory;

            clnf_params = new FaceModelParameters(root, false);
            clnf_model = new CLNF(clnf_params);
            face_analyser = new FaceAnalyserManaged(root, DynamicAUModels, image_output_size);

        }

        // ----------------------------------------------------------
        // Actual work gets done here

        // The main function call for processing images or video files
        private void ProcessingLoop(String[] filenames, int cam_id = -1, int width = -1, int height = -1, bool multi_face = false)
        {

            thread_running = true;

            // Grab the boolean values of the check-boxes
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
                        // Prepare recording if any based on the directory, TODO move this
                        String file_no_ext = System.IO.Path.GetDirectoryName(filenames[0]);
                        file_no_ext = System.IO.Path.GetFileName(file_no_ext);

                        // Start the actual processing                        
                        VideoLoop();                        

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
                            // Prepare recording if any TODO move this
                            String file_no_ext = System.IO.Path.GetFileNameWithoutExtension(filename);
                            
                            // Start the actual processing                        
                            VideoLoop();

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
            if (ShowTrackedVideo)
            {
                Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
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

                }));
            }
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
            double fx = 500.0 * (capture.width / 640.0);
            double fy = 500.0 * (capture.height / 480.0);

            fx = (fx + fy) / 2.0;
            fy = fx;

            double cx = capture.width / 2f;
            double cy = capture.height / 2f;

            // Setup the recorder first, TODO change
            recorder = new Recorder(record_root, "test.txt", capture.width, capture.height, Record2DLandmarks, Record3DLandmarks, RecordModelParameters, RecordPose,
                RecordAUs, RecordGaze, RecordAligned, RecordHOG, clnf_model, face_analyser, fx, fy, cx, cy);

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

                // The face analysis step (for AUs and eye gaze)
                face_analyser.AddNextFrame(frame, clnf_model, fx, fy, cx, cy, false, ShowAppearance, false); // TODO change
                
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
                    if (ShowAUs)
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

                    if (ShowGeometry)
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
                        String x_angle = String.Format("{0:F0}°", gaze_angle.Item1 * (180.0 / Math.PI));
                        String y_angle = String.Format("{0:F0}°", gaze_angle.Item2 * (180.0 / Math.PI));
                        GazeXLabel.Content = x_angle;
                        GazeYLabel.Content = y_angle;
                    }

                    if (ShowTrackedVideo)
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

                    if (ShowAppearance)
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

                recorder.RecordFrame(clnf_model, face_analyser, detectionSucceeding, frame_id + 1, ((double)frame_id) / fps);

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

            recorder.FinishRecording(clnf_model, face_analyser);
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
        // Mode handling (image, video)
        // ----------------------------------------------------------
        private void SetupImageMode()
        {

            // Turn off unneeded visualisations, TODO remove dispatch
            ShowTrackedVideo = true;
            ShowAppearance = false;
            ShowGeometry = false;
            ShowAUs = false;

            RecordAligned = false;
            RecordHOG = false;

            // Actually update the GUI accordingly
            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 2000), (Action)(() =>
            {
                VisualisationChange(null, null);
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

                AUSetting.IsEnabled = true;
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


        private void VisualisationChange(object sender, RoutedEventArgs e)
        {
            // Collapsing or restoring the windows here
            if (!ShowTrackedVideo)
            {
                VideoBorder.Visibility = System.Windows.Visibility.Collapsed;
                MainGrid.ColumnDefinitions[0].Width = new GridLength(0, GridUnitType.Star);
            }
            else
            {
                VideoBorder.Visibility = System.Windows.Visibility.Visible;
                MainGrid.ColumnDefinitions[0].Width = new GridLength(2.1, GridUnitType.Star);
            }

            if (!ShowAppearance)
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
            if (!ShowGeometry)
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
            if (!ShowAUs)
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

        private void UseDynamicModelsCheckBox_Click(object sender, RoutedEventArgs e)
        {
            // Change the face analyser, this should be safe as the model is only allowed to change when not running
            String root = AppDomain.CurrentDomain.BaseDirectory;
            face_analyser = new FaceAnalyserManaged(root, DynamicAUModels, image_output_size);
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
                face_analyser = new FaceAnalyserManaged(root, DynamicAUModels, image_output_size);

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
