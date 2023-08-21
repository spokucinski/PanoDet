using System.Collections.Generic;

namespace DatasetParser.Formats.WiMR
{
    public static class WmirCategoriesDictionary
    {
        public static Dictionary<string, int> WmirCategories { get; set; } = new Dictionary<string, int>()
        {
            { "chair"   , 1  },
            { "tv"      , 2  },
            { "sofa"    , 3  },
            { "cabinet" , 4  },
            { "painting", 5  },
            { "table"   , 6  },
            { "door"    , 7  },
            { "curtain" , 8  },
            { "window"  , 9  },
            { "light"   , 10 },
            {  "shelf"  , 11 },
            {  "mirror" , 12 },
            {  "bedside", 13 },
            {  "bed"    , 14 }
        };
    }
}