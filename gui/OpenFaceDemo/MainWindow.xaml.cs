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

        Thread processing_thread;

        // Some members for displaying the results
        private Capture capture;

        private volatile bool thread_running;

        // For selecting webcams
        CameraSelection cam_sec;

        public MainWindow()
        {
            InitializeComponent();
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

                    // TODO add
                    //processing_thread = new Thread(() => ProcessingLoop(null, cam_id, width, height));
                    //processing_thread.Start();

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

                capture.Dispose();
            }
            // TODO add
            //face_analyser.Dispose();
        }

    }
}
