using Newtonsoft.Json;

namespace DatasetParser.Formats.COCO
{
    public class CocoLicense
    {
        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("url")]
        public string Url { get; set; }
    }
}