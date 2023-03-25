using Newtonsoft.Json;
using System.Collections.Generic;

namespace DatasetParser.Formats.WiMR
{
    public class WmirAnnotation
    {
        [JsonProperty("area")]
        public long Area { get; set; }

        [JsonProperty("bbox")]
        public List<float> Bbox { get; set; }

        [JsonProperty("category_id")]
        public string Category { get; set; }

        [JsonProperty("id")]
        public string Id { get; set; }

        [JsonProperty("image_id")]
        public int ImageId { get; set; }

        [JsonProperty("segmentation")]
        public WmirSegmentation Segmentation { get; set; }
    }
}