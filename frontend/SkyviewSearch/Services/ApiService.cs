using Newtonsoft.Json;
using SkyviewSearch.Models;
using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace SkyviewSearch.Services
{
    
    /// Service for communicating with the Skyview Search API.
    
    public class ApiService : IDisposable
    {
        private readonly HttpClient _client;
        private readonly string _baseUrl;

        public ApiService(string baseUrl = "http://127.0.0.1:8000")
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _client = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(30)
            };
        }

       
        /// Search for images matching the query text.
        
        public async Task<SearchResponse?> SearchAsync(string query, int k = 10)
        {
            try
            {
                var url = $"{_baseUrl}/search?q={Uri.EscapeDataString(query)}&k={k}";
                var response = await _client.GetStringAsync(url);
                return JsonConvert.DeserializeObject<SearchResponse>(response);
            }
            catch (Exception ex)
            {
                throw new ApiException($"Search failed: {ex.Message}", ex);
            }
        }

       
        /// Get database statistics.
       
        public async Task<StatsResponse?> GetStatsAsync()
        {
            try
            {
                var url = $"{_baseUrl}/stats";
                var response = await _client.GetStringAsync(url);
                return JsonConvert.DeserializeObject<StatsResponse>(response);
            }
            catch (Exception ex)
            {
                throw new ApiException($"Failed to get stats: {ex.Message}", ex);
            }
        }

        
        /// Get image as BitmapImage by UUID.
        
        public async Task<BitmapImage?> GetImageAsync(string uuid)
        {
            try
            {
                var url = $"{_baseUrl}/image/{uuid}";
                var bytes = await _client.GetByteArrayAsync(url);

                var bitmap = new BitmapImage();
                using (var stream = new MemoryStream(bytes))
                {
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = stream;
                    bitmap.EndInit();
                    bitmap.Freeze();
                }
                return bitmap;
            }
            catch (Exception ex)
            {
                throw new ApiException($"Failed to get image: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Get thumbnail as BitmapImage by UUID.
        /// </summary>
        public async Task<BitmapImage?> GetThumbnailAsync(string uuid, int size = 200)
        {
            try
            {
                var url = $"{_baseUrl}/thumbnail/{uuid}?size={size}";
                var bytes = await _client.GetByteArrayAsync(url);

                var bitmap = new BitmapImage();
                using (var stream = new MemoryStream(bytes))
                {
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = stream;
                    bitmap.EndInit();
                    bitmap.Freeze();
                }
                return bitmap;
            }
            catch (Exception ex)
            {
                throw new ApiException($"Failed to get thumbnail: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Upload a new image to the database.
        /// </summary>
        public async Task<bool> UploadImageAsync(string filePath, string category = "Uploaded")
        {
            try
            {
                var url = $"{_baseUrl}/upload";
                
                using var form = new MultipartFormDataContent();
                using var fileStream = File.OpenRead(filePath);
                using var streamContent = new StreamContent(fileStream);
                
                streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
                form.Add(streamContent, "file", Path.GetFileName(filePath));
                form.Add(new StringContent(category), "category");

                var response = await _client.PostAsync(url, form);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                throw new ApiException($"Upload failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Check if API is available.
        /// </summary>
        public async Task<bool> IsAvailableAsync()
        {
            try
            {
                var response = await _client.GetAsync($"{_baseUrl}/");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        public void Dispose()
        {
            _client.Dispose();
        }
    }

    /// <summary>
    /// Custom exception for API errors.
    /// </summary>
    public class ApiException : Exception
    {
        public ApiException(string message) : base(message) { }
        public ApiException(string message, Exception inner) : base(message, inner) { }
    }
}
