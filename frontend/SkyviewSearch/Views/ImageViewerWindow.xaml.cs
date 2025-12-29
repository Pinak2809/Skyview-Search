using System.Windows;
using System.Windows.Media.Imaging;

namespace SkyviewSearch.Views
{
    public partial class ImageViewerWindow : Window
    {
        public ImageViewerWindow(BitmapImage image, string caption, string category)
        {
            InitializeComponent();
            
            ImageDisplay.Source = image;
            CaptionText.Text = caption;
            CategoryText.Text = category.ToUpper();
            Title = $"Image Viewer - {category}";
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
