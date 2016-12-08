using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;

// Internal libraries
using OpenFaceOffline;
using OpenCVWrappers;
using CppInterop;
using CppInterop.LandmarkDetector;
using CameraInterop;
using FaceAnalyser_Interop;
using System.Windows.Threading;
using System.Diagnostics;

namespace OpenFaceDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        // -----------------------------------------------------------------
        // Members
        // -----------------------------------------------------------------
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

        Thread processing_thread;

        // Some members for displaying the results
        private Capture capture;
        private WriteableBitmap latest_img;

        private volatile bool thread_running;
        
        FpsTracker processing_fps = new FpsTracker();

        // Controlling the model reset
        volatile bool detectionSucceeding = false;
        volatile bool reset = false;
        Point? resetPoint = null;

        // For selecting webcams
        CameraSelection cam_sec;

        // For tracking
        FaceModelParameters clnf_params;
        CLNF clnf_model;
        FaceAnalyserManaged face_analyser;


        public MainWindow()
        {
            InitializeComponent();

            // Set the icon
            Uri iconUri = new Uri("logo1.ico", UriKind.RelativeOrAbsolute);
            this.Icon = BitmapFrame.Create(iconUri);

            String root = AppDomain.CurrentDomain.BaseDirectory;

            clnf_params = new FaceModelParameters(root, true);
            clnf_model = new CLNF(clnf_params);
            face_analyser = new FaceAnalyserManaged(root, true, 112);

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
            {

                headPosePlot.AssocColor(0, Colors.Blue);
                headPosePlot.AssocColor(1, Colors.Red);
                headPosePlot.AssocColor(2, Colors.Green);

                headPosePlot.AssocName(1, "Turn");
                headPosePlot.AssocName(2, "Tilt");
                headPosePlot.AssocName(0, "Up/Down");

                headPosePlot.AssocThickness(0, 2);
                headPosePlot.AssocThickness(1, 2);
                headPosePlot.AssocThickness(2, 2);

                gazePlot.AssocColor(0, Colors.Red);
                gazePlot.AssocColor(1, Colors.Blue);

                gazePlot.AssocName(0, "Left-right");
                gazePlot.AssocName(1, "Up-down");
                gazePlot.AssocThickness(0, 2);
                gazePlot.AssocThickness(1, 2);

                smilePlot.AssocColor(0, Colors.Green);
                smilePlot.AssocColor(1, Colors.Red);
                smilePlot.AssocName(0, "Smile");
                smilePlot.AssocName(1, "Frown");
                smilePlot.AssocThickness(0, 2);
                smilePlot.AssocThickness(1, 2);
                
                browPlot.AssocColor(0, Colors.Green);
                browPlot.AssocColor(1, Colors.Red);
                browPlot.AssocName(0, "Raise");
                browPlot.AssocName(1, "Furrow");
                browPlot.AssocThickness(0, 2);
                browPlot.AssocThickness(1, 2);

                eyePlot.AssocColor(0, Colors.Green);
                eyePlot.AssocColor(1, Colors.Red);
                eyePlot.AssocName(0, "Eye widen");
                eyePlot.AssocName(1, "Nose wrinkle");
                eyePlot.AssocThickness(0, 2);
                eyePlot.AssocThickness(1, 2);

            }));

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

        // The main function call for processing images, video files or webcam feed
        private void ProcessingLoop(int cam_id = -1, int width = -1, int height = -1, bool multi_face = false)
        {

            thread_running = true;
            
            // Create the video capture from a webcam and call the VideoLoop           
            capture = new Capture(cam_id, width, height);

            if (capture.isOpened())
            {
                // Start the actual processing
                VideoLoop();
            }
            else
            {

                string messageBoxText = "Failed to open a webcam";
                string caption = "Webcam failure";
                MessageBoxButton button = MessageBoxButton.OK;
                MessageBoxImage icon = MessageBoxImage.Warning;

                // Display message box
                MessageBox.Show(messageBoxText, caption, button, icon);
            }

        }

        // Capturing and processing the video frame by frame
        private void VideoLoop()
        {

            Thread.CurrentThread.IsBackground = true;

            DateTime? startTime = CurrentTime;

            var lastFrameTime = CurrentTime;

            clnf_model.Reset();
            face_analyser.Reset();

            double fx, fy, cx, cy;
            fx = 500.0;
            fy = 500.0;
            cx = cy = -1;

            int frame_id = 0;

            double old_gaze_x = 0;
            double old_gaze_y = 0;

            double smile_cumm = 0;
            double frown_cumm = 0;
            double brow_up_cumm = 0;
            double brow_down_cumm = 0;
            double widen_cumm = 0;
            double wrinkle_cumm = 0;

            while (thread_running)
            {
                //////////////////////////////////////////////
                // CAPTURE FRAME AND DETECT LANDMARKS FOLLOWED BY THE REQUIRED IMAGE PROCESSING
                //////////////////////////////////////////////
                RawImage frame = null;
                double progress = -1;

                frame = new RawImage(capture.GetNextFrame(true));
                progress = capture.GetProgress();

                if (frame.Width == 0)
                {
                    // This indicates that we reached the end of the video file
                    break;
                }
                
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

                double confidence = (-clnf_model.GetConfidence()) / 2.0 + 0.5;

                if (confidence < 0)
                    confidence = 0;
                else if (confidence > 1)
                    confidence = 1;

                List<double> pose = new List<double>();

                clnf_model.GetPose(pose, fx, fy, cx, cy);

                List<double> non_rigid_params = clnf_model.GetNonRigidParams();
                double scale = clnf_model.GetRigidParams()[0];

                double time_stamp = (DateTime.Now - (DateTime)startTime).TotalMilliseconds;
                // The face analysis step (only done if recording AUs, HOGs or video)
                face_analyser.AddNextFrame(frame, clnf_model, fx, fy, cx, cy, true, false, false);

                List<Tuple<Point, Point>> lines = null;
                List<Tuple<double, double>> landmarks = null;
                List<Tuple<double, double>> eye_landmarks = null;
                List<Tuple<Point, Point>> gaze_lines = null;
                Tuple<double, double> gaze_angle = face_analyser.GetGazeAngle();

                if (detectionSucceeding)
                {
                    landmarks = clnf_model.CalculateLandmarks();
                    eye_landmarks = clnf_model.CalculateEyeLandmarks();
                    lines = clnf_model.CalculateBox((float)fx, (float)fy, (float)cx, (float)cy);
                    gaze_lines = face_analyser.CalculateGazeLines(scale, (float)fx, (float)fy, (float)cx, (float)cy);
                }

                // Visualisation
                Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
                {

                    var au_regs = face_analyser.GetCurrentAUsReg();

                    double smile = (au_regs["AU12"] + au_regs["AU06"] + au_regs["AU25"]) / 13.0;
                    double frown = (au_regs["AU15"] + au_regs["AU17"]) / 12.0;

                    double brow_up = (au_regs["AU01"] + au_regs["AU02"]) / 10.0;
                    double brow_down = au_regs["AU04"] / 5.0;

                    double eye_widen = au_regs["AU05"] / 3.0;
                    double nose_wrinkle = au_regs["AU09"] / 4.0;

                    Dictionary<int, double> smileDict = new Dictionary<int, double>();
                    smileDict[0] = 0.7 * smile_cumm + 0.3 * smile;
                    smileDict[1] = 0.7 * frown_cumm + 0.3 * frown;
                    smilePlot.AddDataPoint(new DataPointGraph() { Time = CurrentTime, values = smileDict, Confidence = confidence });

                    Dictionary<int, double> browDict = new Dictionary<int, double>();
                    browDict[0] = 0.7 * brow_up_cumm + 0.3 * brow_up;
                    browDict[1] = 0.7 * brow_down_cumm + 0.3 * brow_down;
                    browPlot.AddDataPoint(new DataPointGraph() { Time = CurrentTime, values = browDict, Confidence = confidence });

                    Dictionary<int, double> eyeDict = new Dictionary<int, double>();
                    eyeDict[0] = 0.7 * widen_cumm + 0.3 * eye_widen;
                    eyeDict[1] = 0.7 * wrinkle_cumm + 0.3 * nose_wrinkle;
                    eyePlot.AddDataPoint(new DataPointGraph() { Time = CurrentTime, values = eyeDict, Confidence = confidence });

                    smile_cumm = smileDict[0];
                    frown_cumm = smileDict[1];
                    brow_up_cumm = browDict[0];
                    brow_down_cumm = browDict[1];
                    widen_cumm = eyeDict[0];
                    wrinkle_cumm = eyeDict[1];

                    Dictionary<int, double> poseDict = new Dictionary<int, double>();
                    poseDict[0] = -pose[3];
                    poseDict[1] = pose[4];
                    poseDict[2] = pose[5];
                    headPosePlot.AddDataPoint(new DataPointGraph() { Time = CurrentTime, values = poseDict, Confidence = confidence });

                    Dictionary<int, double> gazeDict = new Dictionary<int, double>();
                    gazeDict[0] = gaze_angle.Item1 * (180.0 / Math.PI);
                    gazeDict[0] = 0.5 * old_gaze_x + 0.5 * gazeDict[0];
                    gazeDict[1] = -gaze_angle.Item2 * (180.0 / Math.PI);
                    gazeDict[1] = 0.5 * old_gaze_y + 0.5 * gazeDict[1];
                    gazePlot.AddDataPoint(new DataPointGraph() { Time = CurrentTime, values = gazeDict, Confidence = confidence });

                    old_gaze_x = gazeDict[0];
                    old_gaze_y = gazeDict[1];
                    
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

                }));

                if (reset)
                {
                    if (resetPoint.HasValue)
                    {
                        clnf_model.Reset(resetPoint.Value.X, resetPoint.Value.Y);
                        resetPoint = null;
                    }
                    else
                    {
                        clnf_model.Reset();
                    }

                    face_analyser.Reset();
                    reset = false;

                    Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 0, 200), (Action)(() =>
                    {
                        headPosePlot.ClearDataPoints();
                        headPosePlot.ClearDataPoints();
                        gazePlot.ClearDataPoints();
                        smilePlot.ClearDataPoints();
                        browPlot.ClearDataPoints();
                        eyePlot.ClearDataPoints();
                    }));
                }

                frame_id++;


            }

            latest_img = null;
        }

        private bool ProcessFrame(CLNF clnf_model, FaceModelParameters clnf_params, RawImage frame, RawImage grayscale_frame, double fx, double fy, double cx, double cy)
        {
            detectionSucceeding = clnf_model.DetectLandmarksInVideo(grayscale_frame, clnf_params);
            return detectionSucceeding;

        }


        // --------------------------------------------------------
        // Button handling
        // --------------------------------------------------------

        private void openWebcamClick(object sender, RoutedEventArgs e)
        {
            new Thread(() => openWebcam()).Start();
        }

        private void openWebcam()
        {
            StopTracking();

            Dispatcher.Invoke(DispatcherPriority.Render, new TimeSpan(0, 0, 0, 2, 0), (Action)(() =>
            {
                // First close the cameras that might be open to avoid clashing with webcam opening
                if (capture != null)
                {
                    capture.Dispose();
                }

                if (cam_sec == null)
                {
                    cam_sec = new CameraSelection(); 
                }
                else
                {
                    cam_sec = new CameraSelection(cam_sec.cams);
                    cam_sec.Visibility = System.Windows.Visibility.Visible;
                }

                // Set the icon
                Uri iconUri = new Uri("logo1.ico", UriKind.RelativeOrAbsolute);
                cam_sec.Icon = BitmapFrame.Create(iconUri);

                if (!cam_sec.no_cameras_found)
                    cam_sec.ShowDialog();

                if (cam_sec.camera_selected)
                {
                    int cam_id = cam_sec.selected_camera.Item1;
                    int width = cam_sec.selected_camera.Item2;
                    int height = cam_sec.selected_camera.Item3;
                    
                    processing_thread = new Thread(() => ProcessingLoop(cam_id, width, height));
                    processing_thread.Start();

                }
            }));
        }



        // Cleanup stuff when closing the window
        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (processing_thread != null)
            {
                // Stop capture and tracking
                thread_running = false;
                processing_thread.Join();

                if (capture != null)
                    capture.Dispose();
                
            }
            if (face_analyser != null)
                face_analyser.Dispose();
            if(clnf_model != null)
                clnf_model.Dispose();

        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.R)
            {
                reset = true;
            }
        }

        private void video_MouseDown(object sender, MouseButtonEventArgs e)
        {
            var clickPos = e.GetPosition(video);
            resetPoint = new Point(clickPos.X / video.ActualWidth, clickPos.Y / video.ActualHeight);
            reset = true;
        }


    }
}
