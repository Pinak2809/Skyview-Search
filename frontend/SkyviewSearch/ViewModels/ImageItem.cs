using SkyviewSearch.Models;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media.Imaging;

namespace SkyviewSearch.ViewModels
{
    /// <summary>
    /// ViewModel for displaying an image result in the UI.
    /// </summary>
    public class ImageItem : INotifyPropertyChanged
    {
        private BitmapImage? _thumbnail;
        private bool _isLoading = true;

        public SearchResult Result { get; }

        public string Uuid => Result.Uuid;
        public string Caption => Result.CaptionShort;
        public string Category => Result.Category ?? "Unknown";
        public string Score => $"{Result.Score:F3}";
        public string? Filepath => Result.Filepath;

        public BitmapImage? Thumbnail
        {
            get => _thumbnail;
            set
            {
                _thumbnail = value;
                OnPropertyChanged();
            }
        }

        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                _isLoading = value;
                OnPropertyChanged();
            }
        }

        public ImageItem(SearchResult result)
        {
            Result = result;
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
