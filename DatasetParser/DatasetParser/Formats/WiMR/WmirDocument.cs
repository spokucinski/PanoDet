using Newtonsoft.Json;
using System.Collections.Generic;

namespace DatasetParser.Formats.WiMR
{
    public class WmirDocument
    {
        [JsonProperty("annotations")]
        public List<WmirAnnotation> Annotations { get; set; }

        [JsonProperty("images")]
        public List<WmirImage> Images { get; set; }

        [JsonProperty("info")]
        public string Info { get; set; }
    }
}