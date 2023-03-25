using Newtonsoft.Json;
using System.Collections.Generic;

namespace DatasetParser.Formats.COCO
{
    public class CocoSegmentation
    {
        [JsonProperty("counts")]
        public List<long> Counts { get; set; }

        [JsonProperty("size")]
        public List<long> Size { get; set; }
    }
}