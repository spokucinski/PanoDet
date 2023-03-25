using Newtonsoft.Json;

namespace DatasetParser.Formats.WiMR
{
    public class WmirImage
    {
        [JsonProperty("file_name")]
        public string FileName { get; set; }

        [JsonProperty("height")]
        public int Height { get; set; }

        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("width")]
        public int Width { get; set; }
    }
}