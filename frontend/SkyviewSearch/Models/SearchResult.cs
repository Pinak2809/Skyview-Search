using Newtonsoft.Json;

namespace SkyviewSearch.Models
{
    /// <summary>
    /// Represents a single image search result from the API.
    /// </summary>
    public class SearchResult
    {
        [JsonProperty("uuid")]
        public string Uuid { get; set; } = string.Empty;

        [JsonProperty("score")]
        public double Score { get; set; }

        [JsonProperty("caption")]
        public string? Caption { get; set; }

        [JsonProperty("category")]
        public string? Category { get; set; }

        [JsonProperty("filepath")]
        public string? Filepath { get; set; }

        /// <summary>
        /// Display score as percentage.
        /// </summary>
        public string ScoreDisplay => $"{Score:P1}";

        /// <summary>
        /// Short caption for display (max 60 chars).
        /// </summary>
        public string CaptionShort => Caption?.Length > 60 
            ? Caption.Substring(0, 57) + "..." 
            : Caption ?? "No caption";
    }
}
