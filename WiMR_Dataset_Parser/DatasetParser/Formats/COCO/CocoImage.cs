using Newtonsoft.Json;

namespace DatasetParser.Formats.COCO
{
    public class CocoImage
    {
        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("width")]
        public int Width { get; set; }

        [JsonProperty("height")]
        public int Height { get; set; }

        [JsonProperty("file_name")]
        public string FileName { get; set; }

        [JsonProperty("license")]
        public int License { get; set; }

        [JsonProperty("flickr_url")]
        public string FlickrUrl { get; set; }

        [JsonProperty("coco_url")]
        public string CocoUrl { get; set; }

        [JsonProperty("date_captured")]
        public string DateCaptured { get; set; }
    }
}