using CppInterop.LandmarkDetector;
using FaceAnalyser_Interop;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenFaceOffline
{
    class Recorder
    {
        StreamWriter output_features_file;

        bool output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, record_aligned, record_HOG;

        double fx, fy, cx, cy;

        List<string> au_reg_names;
        List<string> au_class_names;

        String out_filename;

        bool dynamic_AU_model;

        public Recorder(string root, string filename, int width, int height, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, 
            bool output_pose, bool output_AUs, bool output_gaze, bool record_aligned, bool record_HOG,
            CLNF clnf_model, FaceAnalyserManaged face_analyser, double fx, double fy, double cx, double cy, bool dynamic_AU_model)
        {

            this.output_2D_landmarks = output_2D_landmarks; this.output_3D_landmarks = output_3D_landmarks;
            this.output_model_params = output_model_params; this.output_pose = output_pose;
            this.output_AUs = output_AUs; this.output_gaze = output_gaze;
            this.record_aligned = record_aligned; this.record_HOG = record_HOG;

            this.fx = fx; this.fy = fy; this.cx = cx; this.cy = cy;

            this.dynamic_AU_model = dynamic_AU_model;

            if (!System.IO.Directory.Exists(root))
            {
                System.IO.Directory.CreateDirectory(root);
            }

            // Write out the OF file which tells where all the relevant data is
            StreamWriter out_of_file = new StreamWriter(root + "/" + filename + ".of");

            //out_of_file.WriteLine("Video_file:" + )
            out_of_file.WriteLine("CSV file: " + root + "/" + filename + ".csv");
            if(record_HOG)
            { 
                out_of_file.WriteLine("HOG file: " + root + "/" + filename + ".hog");
            }
            if(record_aligned)
            {
                out_of_file.WriteLine("Aligned dir: " + root + "/" + filename + "/");
            }

            out_filename = root + "/" + filename + ".csv";
            output_features_file = new StreamWriter(out_filename);
            output_features_file.Write("frame, timestamp, confidence, success");

            if (output_gaze)
            {
                output_features_file.Write(", gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y");

                // Output gaze eye landmarks
                int gaze_num_lmks = clnf_model.CalculateEyeLandmarks().Count;
                for (int i = 0; i < gaze_num_lmks; ++i)
                {
                    output_features_file.Write(", eye_lmk_x_" + i);
                }
                for (int i = 0; i < gaze_num_lmks; ++i)
                {
                    output_features_file.Write(", eye_lmk_y_" + i);
                }
            }

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
           
            if (record_HOG)
            {
                String filename_HOG = root + "/" + filename + ".hog";
                face_analyser.SetupHOGRecording(filename_HOG);
            }
        }

        public void RecordFrame(CLNF clnf_model, FaceAnalyserManaged face_analyser, bool success, int frame_ind, double time_stamp)
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

                List<Tuple<double, double>> landmarks_2d = clnf_model.CalculateEyeLandmarks();

                for (int i = 0; i < landmarks_2d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_2d[i].Item1);

                for (int i = 0; i < landmarks_2d.Count; ++i)
                    output_features_file.Write(", {0:F2}", landmarks_2d[i].Item2);

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
            
        }

        public void FinishRecording(CLNF clnf_model, FaceAnalyserManaged face_analyser)
        {
            if (output_features_file != null)
                output_features_file.Close();

            if (record_HOG)
                face_analyser.StopHOGRecording();

            face_analyser.PostProcessOutputFile(out_filename, dynamic_AU_model);
        }

    }
}
