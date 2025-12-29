using Newtonsoft.Json;
using System.Collections.Generic;

namespace SkyviewSearch.Models
{
    /// <summary>
    /// Represents database statistics from the API.
    /// </summary>
    public class StatsResponse
    {
        [JsonProperty("total_images")]
        public int TotalImages { get; set; }

        [JsonProperty("captioned")]
        public int Captioned { get; set; }

        [JsonProperty("embedded")]
        public int Embedded { get; set; }

        [JsonProperty("categories")]
        public Dictionary<string, int> Categories { get; set; } = new();
    }
}
