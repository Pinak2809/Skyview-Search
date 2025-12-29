using SkyviewSearch.ViewModels;
using System.Diagnostics;
using System.Windows;
using System.Windows.Input;

namespace SkyviewSearch.Views
{
    public partial class MainWindow : Window
    {
        private MainViewModel ViewModel => (MainViewModel)DataContext;

        public MainWindow()
        {
            InitializeComponent();
            Closing += MainWindow_Closing;
        }

        /// <summary>
        /// Handle click on image card - open full image viewer.
        /// </summary>
        private async void ImageCard_Click(object sender, MouseButtonEventArgs e)
        {
            if (sender is FrameworkElement element && element.DataContext is ImageItem item)
            {
                // Option 1: Open in default image viewer
                if (!string.IsNullOrEmpty(item.Filepath) && System.IO.File.Exists(item.Filepath))
                {
                    try
                    {
                        Process.Start(new ProcessStartInfo
                        {
                            FileName = item.Filepath,
                            UseShellExecute = true
                        });
                    }
                    catch
                    {
                        // Option 2: Open in built-in viewer
                        await OpenImageViewerAsync(item);
                    }
                }
                else
                {
                    await OpenImageViewerAsync(item);
                }
            }
        }

        /// <summary>
        /// Open image in a new window.
        /// </summary>
        private async System.Threading.Tasks.Task OpenImageViewerAsync(ImageItem item)
        {
            var image = await ViewModel.GetFullImageAsync(item.Uuid);
            if (image != null)
            {
                var viewer = new ImageViewerWindow(image, item.Caption, item.Category);
                viewer.Owner = this;
                viewer.ShowDialog();
            }
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            ViewModel.Dispose();
        }
    }
}
