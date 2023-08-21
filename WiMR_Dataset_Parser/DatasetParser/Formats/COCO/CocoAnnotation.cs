using Newtonsoft.Json;
using System.Collections.Generic;

namespace DatasetParser.Formats.COCO
{
    public class CocoAnnotation
    {
        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("image_id")]
        public int ImageId { get; set; }

        [JsonProperty("category_id")]
        public int CategoryId { get; set; }

        [JsonProperty("segmentation")]
        public CocoSegmentation Segmentation { get; set; }

        [JsonProperty("area")]
        public float Area { get; set; }

        [JsonProperty("bbox")]
        public List<float> BBox { get; set; }

        [JsonProperty("iscrowd")]
        public int IsCrowd { get; set; }
    }
}