using System.Windows.Controls;

namespace OpenFaceOffline
{
    /// <summary>
    /// Interaction logic for BarGraph.xaml
    /// </summary>
    public partial class BarGraph : UserControl
    {
        private double targetValue = 0;

        public BarGraph()
        {
            InitializeComponent();
        }

        public void SetValue(double value)
        {
            targetValue = 1.5 * value;
            if (targetValue > 0)
            {
                if (targetValue > barContainerPos.ActualHeight)
                    targetValue = barContainerPos.ActualHeight;

                barPos.Height = targetValue;
                barNeg.Height = 0;
            }
            if (targetValue < 0)
            {
                if (-targetValue > barContainerNeg.ActualHeight)
                    targetValue = -barContainerNeg.ActualHeight;

                barPos.Height = 0;
                barNeg.Height = -targetValue;
            }
        }

    }
}
