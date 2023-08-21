using Newtonsoft.Json;

namespace DatasetParser.Formats.COCO
{
    public class CocoCategory
    {
        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("supercategory")]
        public string SuperCategory { get; set; }
    }
}