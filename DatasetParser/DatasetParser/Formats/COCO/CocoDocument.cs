using Newtonsoft.Json;
using System.Collections.Generic;

namespace DatasetParser.Formats.COCO
{
    public class CocoDocument
    {
        [JsonProperty("licenses")]
        public List<CocoLicense> Licenses { get; set; }

        [JsonProperty("info")]
        public CocoInfo Info { get; set; }

        [JsonProperty("categories")]
        public List<CocoCategory> Categories { get; set; }

        [JsonProperty("images")]
        public List<CocoImage> Images { get; set; }

        [JsonProperty("annotations")]
        public List<CocoAnnotation> Annotations { get; set; }
    }
}