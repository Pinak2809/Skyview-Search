using SkyviewSearch.Services;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace SkyviewSearch.ViewModels
{
    /// <summary>
    /// Main ViewModel for the application.
    /// </summary>
    public class MainViewModel : INotifyPropertyChanged, IDisposable
    {
        private readonly ApiService _api;
        private string _searchQuery = string.Empty;
        private string _statusMessage = "Ready";
        private bool _isSearching;
        private bool _isConnected;
        private int _totalImages;
        private int _resultCount = 10;

        public ObservableCollection<ImageItem> SearchResults { get; } = new();

        public string SearchQuery
        {
            get => _searchQuery;
            set
            {
                _searchQuery = value;
                OnPropertyChanged();
            }
        }

        public string StatusMessage
        {
            get => _statusMessage;
            set
            {
                _statusMessage = value;
                OnPropertyChanged();
            }
        }

        public bool IsSearching
        {
            get => _isSearching;
            set
            {
                _isSearching = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(CanSearch));
            }
        }

        public bool IsConnected
        {
            get => _isConnected;
            set
            {
                _isConnected = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(ConnectionStatus));
            }
        }

        public string ConnectionStatus => IsConnected ? "● Connected" : "○ Disconnected";

        public int TotalImages
        {
            get => _totalImages;
            set
            {
                _totalImages = value;
                OnPropertyChanged();
            }
        }

        public int ResultCount
        {
            get => _resultCount;
            set
            {
                _resultCount = Math.Clamp(value, 1, 50);
                OnPropertyChanged();
            }
        }

        public bool CanSearch => !IsSearching && IsConnected && !string.IsNullOrWhiteSpace(SearchQuery);

        public ICommand SearchCommand { get; }
        public ICommand ClearCommand { get; }

        public MainViewModel()
        {
            _api = new ApiService();
            SearchCommand = new RelayCommand(async _ => await SearchAsync(), _ => CanSearch);
            ClearCommand = new RelayCommand(_ => ClearResults());
            
            // Check connection on startup
            Task.Run(CheckConnectionAsync);
        }

        /// <summary>
        /// Check if the API is available.
        /// </summary>
        public async Task CheckConnectionAsync()
        {
            try
            {
                IsConnected = await _api.IsAvailableAsync();
                
                if (IsConnected)
                {
                    var stats = await _api.GetStatsAsync();
                    if (stats != null)
                    {
                        TotalImages = stats.TotalImages;
                        StatusMessage = $"Connected - {TotalImages:N0} images in database";
                    }
                }
                else
                {
                    StatusMessage = "API not available. Start the backend server.";
                }
            }
            catch (Exception ex)
            {
                IsConnected = false;
                StatusMessage = $"Connection error: {ex.Message}";
            }
        }

        /// <summary>
        /// Execute search query.
        /// </summary>
        public async Task SearchAsync()
        {
            if (string.IsNullOrWhiteSpace(SearchQuery))
                return;

            IsSearching = true;
            StatusMessage = $"Searching for '{SearchQuery}'...";
            SearchResults.Clear();

            try
            {
                var response = await _api.SearchAsync(SearchQuery, ResultCount);
                
                if (response?.Results != null)
                {
                    foreach (var result in response.Results)
                    {
                        var item = new ImageItem(result);
                        SearchResults.Add(item);
                        
                        // Load thumbnail asynchronously
                        _ = LoadThumbnailAsync(item);
                    }
                    
                    StatusMessage = $"Found {response.Count} results for '{SearchQuery}'";
                }
                else
                {
                    StatusMessage = "No results found";
                }
            }
            catch (ApiException ex)
            {
                StatusMessage = ex.Message;
                MessageBox.Show(ex.Message, "Search Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
            }
            finally
            {
                IsSearching = false;
            }
        }

        /// <summary>
        /// Load thumbnail for an image item.
        /// </summary>
        private async Task LoadThumbnailAsync(ImageItem item)
        {
            try
            {
                var thumbnail = await _api.GetThumbnailAsync(item.Uuid, 200);
                
                // Update on UI thread
                Application.Current.Dispatcher.Invoke(() =>
                {
                    item.Thumbnail = thumbnail;
                    item.IsLoading = false;
                });
            }
            catch
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    item.IsLoading = false;
                });
            }
        }

        /// <summary>
        /// Clear search results.
        /// </summary>
        public void ClearResults()
        {
            SearchResults.Clear();
            SearchQuery = string.Empty;
            StatusMessage = $"Ready - {TotalImages:N0} images in database";
        }

        /// <summary>
        /// Get full image for viewing.
        /// </summary>
        public async Task<System.Windows.Media.Imaging.BitmapImage?> GetFullImageAsync(string uuid)
        {
            try
            {
                return await _api.GetImageAsync(uuid);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to load image: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return null;
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
            
            // Refresh CanSearch when related properties change
            if (name == nameof(SearchQuery))
            {
                CommandManager.InvalidateRequerySuggested();
            }
        }

        public void Dispose()
        {
            _api.Dispose();
        }
    }

    /// <summary>
    /// Simple ICommand implementation.
    /// </summary>
    public class RelayCommand : ICommand
    {
        private readonly Action<object?> _execute;
        private readonly Func<object?, bool>? _canExecute;

        public RelayCommand(Action<object?> execute, Func<object?, bool>? canExecute = null)
        {
            _execute = execute;
            _canExecute = canExecute;
        }

        public bool CanExecute(object? parameter) => _canExecute?.Invoke(parameter) ?? true;

        public void Execute(object? parameter) => _execute(parameter);

        public event EventHandler? CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
