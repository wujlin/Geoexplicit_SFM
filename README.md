这是一个基于geo-explicit并且融合diffusion增强的模型，目标关注Monroe, Macomb, Oakland, St. Clair, Livingston, Washtenaw, and Wayne这几个地方的od flow。

主要的数据集放在dataset文件夹下，包括geo的数据，主要是geo/mi-tracts-demo-work，它里面有每个tract的地理边界信息，当然也有home-work的od flow信息od_flow/mi-tract-od-2020。路网数据主要放在dataset\geo\MI_road_cleaned.shp。详细的信息你可以读一下具体的文件。