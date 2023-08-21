using Newtonsoft.Json;

namespace DatasetParser.Formats.COCO
{
    public class CocoInfo
    {
        [JsonProperty("contributor")]
        public string Contributor { get; set; }

        [JsonProperty("date_created")]
        public string CreationDate { get; set; }

        [JsonProperty("description")]
        public string Description { get; set; }

        [JsonProperty("url")]
        public string Url { get; set; }

        [JsonProperty("version")]
        public string Version { get; set; }

        [JsonProperty("year")]
        public string Year { get; set; }
    }
}