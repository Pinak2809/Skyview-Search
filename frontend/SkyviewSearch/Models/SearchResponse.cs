using Newtonsoft.Json;
using System.Collections.Generic;

namespace SkyviewSearch.Models
{
    
    /// Represents the API response for a search query.
    
    public class SearchResponse
    {
        [JsonProperty("query")]
        public string Query { get; set; } = string.Empty;

        [JsonProperty("k")]
        public int K { get; set; }

        [JsonProperty("count")]
        public int Count { get; set; }

        [JsonProperty("results")]
        public List<SearchResult> Results { get; set; } = new();
    }
}
